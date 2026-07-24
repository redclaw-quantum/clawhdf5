//! SZIP (libaec Adaptive Entropy Coding) decompression.
//!
//! Gated by the `szip` feature which links against the system libaec library.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::FormatError;

/// Decompress SZIP-compressed data using libaec.
///
/// `cd` is the HDF5 SZIP filter client data (matches `H5Z_SZIP_PARM_*` indices):
///   cd[0] = options mask  (`H5_SZIP_NN_OPTION_MASK = 0x20` enables NN preprocessing)
///   cd[1] = pixels per block (H5Z_SZIP_PARM_PPB; 8, 10, 16, or 32)
///   cd[2] = bits per sample  (H5Z_SZIP_PARM_BPP; element bit width)
///   cd[3] = pixels per scan line (H5Z_SZIP_PARM_PPS; informational only)
pub(crate) fn szip_decompress(
    _data: &[u8],
    _cd: &[u32],
    _chunk_size: usize,
) -> Result<Vec<u8>, FormatError> {
    #[cfg(feature = "szip")]
    {
        szip_decode_impl(_data, _cd, _chunk_size)
    }
    #[cfg(not(feature = "szip"))]
    {
        Err(FormatError::UnsupportedFilter(
            crate::filter_pipeline::FILTER_SZIP,
        ))
    }
}

#[cfg(feature = "szip")]
fn szip_decode_impl(data: &[u8], cd: &[u32], chunk_size: usize) -> Result<Vec<u8>, FormatError> {
    if cd.len() < 3 {
        return Err(FormatError::ChunkedReadError(
            "szip: missing client data".into(),
        ));
    }
    let options = cd[0];
    let pixels_per_block = cd[1];
    let bits_per_sample = cd[2]; // H5Z_SZIP_PARM_BPP
    if bits_per_sample == 0 || bits_per_sample > 32 {
        return Err(FormatError::ChunkedReadError(
            "szip: invalid bits per sample".into(),
        ));
    }
    if chunk_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "szip: unknown output size".into(),
        ));
    }
    if data.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "szip: empty input".into(),
        ));
    }

    // Map HDF5 option mask to libaec flags.
    // HDF5 always stores SZIP data in MSB order, so AEC_DATA_MSB is unconditional.
    // H5_SZIP_NN_OPTION_MASK (0x20): NN differential preprocessing.
    let mut flags: u32 = libaec_sys::AEC_DATA_MSB;
    if options & 0x20 != 0 {
        flags |= libaec_sys::AEC_DATA_PREPROCESS;
    }

    let mut out = vec![0u8; chunk_size];
    let mut strm = libaec_sys::AecStream::zeroed();
    strm.next_in = data.as_ptr();
    strm.avail_in = data.len();
    strm.next_out = out.as_mut_ptr();
    strm.avail_out = chunk_size;
    strm.bits_per_sample = bits_per_sample;
    strm.block_size = pixels_per_block;
    strm.rsi = 128; // HDF5 default: 128 blocks per reference sample interval
    strm.flags = flags;

    let result = unsafe { libaec_sys::aec_buffer_decode(&mut strm) };
    if result != 0 {
        return Err(FormatError::DecompressionError(format!(
            "szip: libaec error {result}"
        )));
    }
    let decoded_len = chunk_size - strm.avail_out;
    out.truncate(decoded_len);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn szip_disabled_returns_unsupported() {
        #[cfg(not(feature = "szip"))]
        {
            let result = szip_decompress(&[], &[0, 8, 8, 1024], 64);
            assert!(
                matches!(result, Err(FormatError::UnsupportedFilter(4))),
                "expected UnsupportedFilter(4), got {result:?}"
            );
        }
        #[cfg(feature = "szip")]
        {
            // When szip IS enabled, an empty buffer should error but not panic.
            let result = szip_decompress(&[], &[0, 8, 8, 1024], 64);
            assert!(result.is_err(), "empty buffer must not succeed");
        }
    }

    /// Round-trip test: encode with libaec then decode through szip_decompress.
    ///
    /// Uses 1024 samples (rsi=128 × block_size=8) so the block count is exact.
    #[cfg(feature = "szip")]
    #[test]
    fn roundtrip_u8_msb_no_nn() {
        use libaec_sys::{AecStream, AEC_DATA_MSB};

        let original: Vec<u8> = (0..1024u32).map(|i| (i % 256) as u8).collect();

        // Encode with libaec directly (no NN, MSB — mirrors what HDF5 always writes).
        let mut encoded = vec![0u8; original.len() * 2];
        let mut enc = AecStream::zeroed();
        enc.next_in = original.as_ptr();
        enc.avail_in = original.len();
        enc.next_out = encoded.as_mut_ptr();
        enc.avail_out = encoded.len();
        enc.bits_per_sample = 8;
        enc.block_size = 8;
        enc.rsi = 128;
        enc.flags = AEC_DATA_MSB;
        let rc = unsafe { libaec_sys::aec_buffer_encode(&mut enc) };
        assert_eq!(rc, 0, "aec_buffer_encode failed: {rc}");
        let enc_len = encoded.len() - enc.avail_out;
        encoded.truncate(enc_len);

        // Decode through our public interface.
        // cd[0]=0 (no NN bit 0x20), cd[1]=8 (ppb), cd[2]=8 (bpp), cd[3]=1024 (pps).
        let cd = [0u32, 8, 8, 1024];
        let decoded = szip_decompress(&encoded, &cd, original.len())
            .expect("szip_decompress must succeed on valid libaec output");
        assert_eq!(decoded, original, "round-trip must reproduce original data");
    }

    /// Same round-trip but with NN preprocessing enabled (H5_SZIP_NN_OPTION_MASK = 0x20).
    #[cfg(feature = "szip")]
    #[test]
    fn roundtrip_u8_msb_with_nn() {
        use libaec_sys::{AecStream, AEC_DATA_MSB, AEC_DATA_PREPROCESS};

        let original: Vec<u8> = (0..1024u32).map(|i| (i % 256) as u8).collect();

        let mut encoded = vec![0u8; original.len() * 2];
        let mut enc = AecStream::zeroed();
        enc.next_in = original.as_ptr();
        enc.avail_in = original.len();
        enc.next_out = encoded.as_mut_ptr();
        enc.avail_out = encoded.len();
        enc.bits_per_sample = 8;
        enc.block_size = 8;
        enc.rsi = 128;
        enc.flags = AEC_DATA_MSB | AEC_DATA_PREPROCESS;
        let rc = unsafe { libaec_sys::aec_buffer_encode(&mut enc) };
        assert_eq!(rc, 0, "aec_buffer_encode with NN failed: {rc}");
        let enc_len = encoded.len() - enc.avail_out;
        encoded.truncate(enc_len);

        // cd[0] = 0x20 (H5_SZIP_NN_OPTION_MASK) → decoder must set AEC_DATA_PREPROCESS.
        let cd = [0x20u32, 8, 8, 1024];
        let decoded = szip_decompress(&encoded, &cd, original.len())
            .expect("szip_decompress with NN must succeed");
        assert_eq!(decoded, original, "NN round-trip must reproduce original data");
    }
}
