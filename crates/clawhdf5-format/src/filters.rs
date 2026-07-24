//! HDF5 filter implementations: deflate, shuffle, fletcher32.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::error::FormatError;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZ4, FILTER_NBIT, FILTER_PCODEC,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, FILTER_ZSTD, FilterPipeline,
};

/// Apply a filter pipeline to decompress a chunk.
/// Filters are applied in REVERSE order for decompression.
pub fn decompress_chunk(
    compressed: &[u8],
    pipeline: &FilterPipeline,
    chunk_size: usize,
    element_size: u32,
) -> Result<Vec<u8>, FormatError> {
    let mut data = compressed.to_vec();

    for filter in pipeline.filters.iter().rev() {
        data = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_decompress(&data, element_size as usize)?,
            FILTER_DEFLATE => deflate_decompress(&data)?,
            FILTER_LZ4 => lz4_decompress(&data)?,
            FILTER_ZSTD => zstd_decompress(&data)?,
            FILTER_FLETCHER32 => fletcher32_verify(&data)?,
            FILTER_PCODEC => pcodec_decompress(&data, element_size as usize)?,
            // `chunk_size` is the expected decompressed size; pass it so these
            // decoders can reject an element count that would over-allocate.
            FILTER_SCALEOFFSET => scaleoffset_decompress(&data, &filter.client_data, chunk_size)?,
            FILTER_NBIT => nbit_decompress(&data, &filter.client_data, chunk_size)?,
            FILTER_SZIP => crate::filters_szip::szip_decompress(&data, &filter.client_data, chunk_size)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
    }

    Ok(data)
}

/// Apply a filter pipeline to compress a chunk.
/// Filters are applied in FORWARD order for compression.
pub fn compress_chunk(
    data: &[u8],
    pipeline: &FilterPipeline,
    element_size: u32,
) -> Result<Vec<u8>, FormatError> {
    let mut result = data.to_vec();

    for filter in &pipeline.filters {
        result = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_compress(&result, element_size as usize)?,
            FILTER_DEFLATE => {
                let level = filter.client_data.first().copied().unwrap_or(6);
                deflate_compress(&result, level)?
            }
            FILTER_LZ4 => lz4_compress(&result)?,
            FILTER_ZSTD => {
                let level = filter.client_data.first().copied().unwrap_or(3);
                zstd_compress(&result, level)?
            }
            FILTER_FLETCHER32 => fletcher32_append(&result)?,
            FILTER_PCODEC => pcodec_compress(&result, element_size as usize)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
    }

    Ok(result)
}

/// Decode the HDF5 scale-offset filter (id 6).
///
/// Supports all three scale-offset variants:
/// - `H5Z_SO_FLOAT_DSCALE` (0): `value = minval + code / 10^D`
/// - `H5Z_SO_FLOAT_ESCALE` (1): `value = minval + code * 2^E`
/// - `H5Z_SO_INT` (2): `value = minval + code`
///
/// Compressed buffer layout: `minbits` (u32 LE) · `minval_width` (1 byte)
/// · `minval` (`minval_width` bytes) · 8 reserved bytes · MSB-first packed
/// codes (`nelmts * minbits` bits). The all-ones code is reserved for the
/// defined fill value.
///
/// `cd` is the `H5Zscaleoffset.c` parameter block: `[0]`=scale type,
/// `[1]`=scale factor (decimal digits D for D-scale, binary exponent E for
/// E-scale, interpreted as i32 for negative exponents), `[2]`=element count,
/// `[4]`=element size, `[5]`=signed flag, `[6]`=byte order (1 = big-endian),
/// `[7]`=fill defined, `[8..]`=fill value bits.
fn scaleoffset_decompress(
    data: &[u8],
    cd: &[u32],
    expected_bytes: usize,
) -> Result<Vec<u8>, FormatError> {
    const H5Z_SO_FLOAT_DSCALE: u32 = 0;
    const H5Z_SO_FLOAT_ESCALE: u32 = 1;
    const H5Z_SO_INT: u32 = 2;
    if cd.len() < 8 {
        return Err(FormatError::ChunkedReadError(
            "scale-offset: missing filter client data".into(),
        ));
    }
    let scale_type = cd[0];
    let is_float = scale_type == H5Z_SO_FLOAT_DSCALE || scale_type == H5Z_SO_FLOAT_ESCALE;
    if scale_type != H5Z_SO_INT && !is_float {
        return Err(FormatError::UnsupportedFilter(FILTER_SCALEOFFSET));
    }
    let nelmts = cd[2] as usize;
    let elem_size = cd[4] as usize;
    if elem_size == 0 || elem_size > 8 || (is_float && elem_size != 4 && elem_size != 8) {
        return Err(FormatError::ChunkedReadError(
            "scale-offset: unsupported element size".into(),
        ));
    }
    // The decoded output must match the chunk's uncompressed size; reject an
    // element count that would over-allocate (e.g. minbits == 0 with a huge
    // nelmts and no packed payload to bound it).
    let out_bytes = nelmts
        .checked_mul(elem_size)
        .ok_or_else(|| FormatError::ChunkedReadError("scale-offset: size overflow".into()))?;
    if expected_bytes != 0 && out_bytes > expected_bytes {
        return Err(FormatError::ChunkedReadError(
            "scale-offset: element count exceeds chunk size".into(),
        ));
    }
    let signed = cd[5] == 1;
    let big_endian = cd[6] == 1;
    let fill_defined = cd[7] == 1;

    // --- header: minbits, then minval, then 8 reserved bytes ---
    if data.len() < 5 {
        return Err(FormatError::ChunkedReadError(
            "scale-offset: truncated header".into(),
        ));
    }
    let minbits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let minval_width = data[4] as usize;
    let minval_end = 5 + minval_width;
    if data.len() < minval_end {
        return Err(FormatError::ChunkedReadError(
            "scale-offset: truncated minval".into(),
        ));
    }
    let minval_bytes = &data[5..minval_end];

    // --- unpack the per-element codes (MSB-first), shared by both variants ---
    if minbits > 64 {
        return Err(FormatError::ChunkedReadError(
            "scale-offset: implausible minbits".into(),
        ));
    }
    let codes: Vec<u64> = if minbits == 0 {
        // No packed payload: every element equals minval.
        vec![0u64; nelmts]
    } else {
        let packed = data.get(minval_end + 8..).ok_or_else(|| {
            FormatError::ChunkedReadError("scale-offset: truncated packed data".into())
        })?;
        let need_bits = nelmts
            .checked_mul(minbits)
            .ok_or_else(|| FormatError::ChunkedReadError("scale-offset: size overflow".into()))?;
        if packed.len() * 8 < need_bits {
            return Err(FormatError::ChunkedReadError(
                "scale-offset: packed data too short".into(),
            ));
        }
        let mut out = Vec::with_capacity(nelmts);
        let mut bitpos = 0usize;
        for _ in 0..nelmts {
            let mut code: u64 = 0;
            for _ in 0..minbits {
                let bit = (packed[bitpos / 8] >> (7 - (bitpos % 8))) & 1;
                code = (code << 1) | bit as u64;
                bitpos += 1;
            }
            out.push(code);
        }
        out
    };
    // The fill code (all ones) only exists when there are bits to pack.
    let has_fill_code = fill_defined && minbits > 0 && minbits < 64;
    // Computed for all 1..=64 widths; `1 << 64` would overflow, so saturate.
    let fill_code: u64 = if minbits == 0 {
        0
    } else if minbits >= 64 {
        u64::MAX
    } else {
        (1u64 << minbits) - 1
    };

    if is_float {
        let is_escale = scale_type == H5Z_SO_FLOAT_ESCALE;
        let scale_factor = cd[1] as i32;
        let minval = read_le_float(minval_bytes, elem_size);
        let fill_value = if fill_defined {
            let lo = *cd.get(8).unwrap_or(&0) as u64;
            let hi = *cd.get(9).unwrap_or(&0) as u64;
            bits_to_float(lo | (hi << 32), elem_size)
        } else {
            0.0
        };
        let values: Vec<f64> = codes
            .iter()
            .map(|&code| {
                if has_fill_code && code == fill_code {
                    fill_value
                } else if is_escale {
                    minval + code as f64 * powi_bounded(2.0, scale_factor)
                } else {
                    minval + code as f64 / powi_bounded(10.0, scale_factor)
                }
            })
            .collect();
        Ok(write_floats(&values, elem_size, big_endian))
    } else {
        let minval = read_le_int(minval_bytes, signed);
        let fill_value: i64 = if fill_defined {
            let lo = *cd.get(8).unwrap_or(&0) as u64;
            let hi = *cd.get(9).unwrap_or(&0) as u64;
            sign_extend(lo | (hi << 32), elem_size, signed)
        } else {
            0
        };
        let values: Vec<i64> = codes
            .iter()
            .map(|&code| {
                if has_fill_code && code == fill_code {
                    fill_value
                } else {
                    minval.wrapping_add(code as i64)
                }
            })
            .collect();
        Ok(write_elements(&values, elem_size, big_endian))
    }
}

/// Integer power for scale-offset decoding (D-scale base 10, E-scale base 2).
///
/// `f64::powi` lives in `std`, not `core`, so this crate computes it by
/// repeated multiplication to stay `no_std`-clean.
///
/// The exponent comes from the file's filter client data and is therefore
/// attacker-controlled: the loop exits as soon as the accumulator saturates
/// to infinity, so a hostile exponent cannot buy unbounded work.
fn powi_bounded(base: f64, exp: i32) -> f64 {
    let mut result = 1.0f64;
    for _ in 0..exp.unsigned_abs() {
        result *= base;
        if result.is_infinite() {
            break;
        }
    }
    if exp < 0 { 1.0 / result } else { result }
}

/// Read a little-endian float of `size` bytes (4 = f32, otherwise f64) as f64.
fn read_le_float(bytes: &[u8], size: usize) -> f64 {
    if size == 4 {
        let mut b = [0u8; 4];
        let n = bytes.len().min(4);
        b[..n].copy_from_slice(&bytes[..n]);
        f32::from_le_bytes(b) as f64
    } else {
        let mut b = [0u8; 8];
        let n = bytes.len().min(8);
        b[..n].copy_from_slice(&bytes[..n]);
        f64::from_le_bytes(b)
    }
}

/// Interpret the low bits of `raw` as an IEEE float of `size` bytes.
fn bits_to_float(raw: u64, size: usize) -> f64 {
    if size == 4 {
        f32::from_bits(raw as u32) as f64
    } else {
        f64::from_bits(raw)
    }
}

/// Serialize reconstructed float values as `elem_size`-byte (f32/f64) elements
/// in the requested byte order.
fn write_floats(values: &[f64], elem_size: usize, big_endian: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * elem_size);
    for &v in values {
        let bytes: [u8; 8] = if elem_size == 4 {
            let mut b = [0u8; 8];
            b[..4].copy_from_slice(&(v as f32).to_le_bytes());
            b
        } else {
            v.to_le_bytes()
        };
        if big_endian {
            for i in (0..elem_size).rev() {
                out.push(bytes[i]);
            }
        } else {
            out.extend_from_slice(&bytes[..elem_size]);
        }
    }
    out
}

/// Read a little-endian integer of `bytes.len()` bytes, sign-extending when
/// `signed`. Used for the scale-offset `minval` field.
fn read_le_int(bytes: &[u8], signed: bool) -> i64 {
    let mut raw: u64 = 0;
    for (i, &b) in bytes.iter().enumerate().take(8) {
        raw |= (b as u64) << (i * 8);
    }
    sign_extend(raw, bytes.len().min(8), signed)
}

/// Interpret the low `size` bytes of `raw` as a (possibly signed) integer.
fn sign_extend(raw: u64, size: usize, signed: bool) -> i64 {
    if size == 0 || size >= 8 {
        return raw as i64;
    }
    let bits = size * 8;
    let mask = (1u64 << bits) - 1;
    let val = raw & mask;
    if signed && (val & (1u64 << (bits - 1))) != 0 {
        (val | !mask) as i64
    } else {
        val as i64
    }
}

/// Serialize reconstructed integer values as `elem_size`-byte elements in the
/// requested byte order.
fn write_elements(values: &[i64], elem_size: usize, big_endian: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * elem_size);
    for &v in values {
        let le = (v as u64).to_le_bytes();
        if big_endian {
            for i in (0..elem_size).rev() {
                out.push(le[i]);
            }
        } else {
            out.extend_from_slice(&le[..elem_size]);
        }
    }
    out
}

/// A node of the N-Bit datatype tree (reconstructed from the filter's client
/// data) describing how one element is bit-packed.
enum NbitNode {
    /// Leaf: `precision` significant bits at `bit_offset` of a `size`-byte field.
    Atomic {
        size: usize,
        big_endian: bool,
        precision: u32,
        bit_offset: u32,
    },
    /// Fixed-size struct of members, each at a byte offset within the element.
    Compound {
        size: usize,
        members: Vec<(usize, NbitNode)>,
    },
    /// `count` consecutive copies of `base`, each `base_size` bytes apart.
    Array {
        base: Box<NbitNode>,
        count: usize,
        base_size: usize,
    },
}

impl NbitNode {
    fn byte_size(&self) -> usize {
        match self {
            NbitNode::Atomic { size, .. } => *size,
            NbitNode::Compound { size, .. } => *size,
            NbitNode::Array {
                count, base_size, ..
            } => count * base_size,
        }
    }
}

fn nbit_cd(cd: &[u32], i: usize) -> Result<u32, FormatError> {
    cd.get(i)
        .copied()
        .ok_or_else(|| FormatError::ChunkedReadError("nbit: truncated client data".into()))
}

/// Maximum N-Bit type-tree nesting depth. Real types nest only a few levels;
/// the bound stops a crafted tree from recursing into a stack overflow.
const NBIT_MAX_DEPTH: u32 = 64;

/// Parse one N-Bit type node from the client-data tree, advancing `idx`.
fn parse_nbit_node(cd: &[u32], idx: &mut usize, depth: u32) -> Result<NbitNode, FormatError> {
    const ATOMIC: u32 = 1;
    const ARRAY: u32 = 2;
    const COMPOUND: u32 = 3;
    if depth > NBIT_MAX_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "nbit: type tree nested too deeply".into(),
        ));
    }
    let class = nbit_cd(cd, *idx)?;
    match class {
        ATOMIC => {
            // class, size, byte order, precision, bit offset
            let size = nbit_cd(cd, *idx + 1)? as usize;
            let big_endian = nbit_cd(cd, *idx + 2)? == 1;
            let precision = nbit_cd(cd, *idx + 3)?;
            let bit_offset = nbit_cd(cd, *idx + 4)?;
            *idx += 5;
            let end_bit = bit_offset.checked_add(precision);
            if size == 0
                || size > 8
                || precision == 0
                || end_bit.is_none_or(|e| e > (size * 8) as u32)
            {
                return Err(FormatError::ChunkedReadError(
                    "nbit: invalid atomic parameters".into(),
                ));
            }
            Ok(NbitNode::Atomic {
                size,
                big_endian,
                precision,
                bit_offset,
            })
        }
        ARRAY => {
            // class, total size, base type node
            let total = nbit_cd(cd, *idx + 1)? as usize;
            *idx += 2;
            let base = parse_nbit_node(cd, idx, depth + 1)?;
            let base_size = base.byte_size();
            if base_size == 0 || !total.is_multiple_of(base_size) {
                return Err(FormatError::ChunkedReadError(
                    "nbit: invalid array layout".into(),
                ));
            }
            Ok(NbitNode::Array {
                count: total / base_size,
                base_size,
                base: Box::new(base),
            })
        }
        COMPOUND => {
            // class, total size, member count, (member byte offset, node)*
            let total = nbit_cd(cd, *idx + 1)? as usize;
            let nmembers = nbit_cd(cd, *idx + 2)? as usize;
            *idx += 3;
            // Don't pre-allocate from the untrusted member count; the loop is
            // bounded by `nbit_cd` running out of client data.
            let mut members = Vec::new();
            for _ in 0..nmembers {
                let moff = nbit_cd(cd, *idx)? as usize;
                *idx += 1;
                let node = parse_nbit_node(cd, idx, depth + 1)?;
                if moff
                    .checked_add(node.byte_size())
                    .is_none_or(|end| end > total)
                {
                    return Err(FormatError::ChunkedReadError(
                        "nbit: member exceeds compound size".into(),
                    ));
                }
                members.push((moff, node));
            }
            Ok(NbitNode::Compound {
                size: total,
                members,
            })
        }
        // Class 4 is H5Z_NBIT_NOOPTYPE (members copied verbatim) — not seen in
        // practice for the supported leaf types and left unsupported.
        _ => Err(FormatError::UnsupportedFilter(FILTER_NBIT)),
    }
}

/// MSB-first bit reader over the packed N-Bit stream.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn read(&mut self, nbits: u32) -> Result<u64, FormatError> {
        let mut value = 0u64;
        for _ in 0..nbits {
            let byte = *self.data.get(self.pos / 8).ok_or_else(|| {
                FormatError::ChunkedReadError("nbit: packed data too short".into())
            })?;
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            value = (value << 1) | bit as u64;
            self.pos += 1;
        }
        Ok(value)
    }
}

/// Decode one element node into `elem[base..]` (the rest stays zero-filled).
fn decode_nbit_node(
    node: &NbitNode,
    br: &mut BitReader,
    elem: &mut [u8],
    base: usize,
) -> Result<(), FormatError> {
    match node {
        NbitNode::Atomic {
            size,
            big_endian,
            precision,
            bit_offset,
        } => {
            let value = br.read(*precision)? << bit_offset;
            let le = value.to_le_bytes();
            let dst = &mut elem[base..base + size];
            if *big_endian {
                for (j, slot) in dst.iter_mut().enumerate() {
                    *slot = le[size - 1 - j];
                }
            } else {
                dst.copy_from_slice(&le[..*size]);
            }
        }
        NbitNode::Compound { members, .. } => {
            for (moff, child) in members {
                decode_nbit_node(child, br, elem, base + moff)?;
            }
        }
        NbitNode::Array {
            base: bnode,
            count,
            base_size,
        } => {
            for i in 0..*count {
                decode_nbit_node(bnode, br, elem, base + i * base_size)?;
            }
        }
    }
    Ok(())
}

/// Decode the HDF5 N-Bit filter (id 5).
///
/// N-Bit strips the unused leading/trailing bits of each (possibly nested)
/// datatype field and packs the significant `precision` bits MSB-first,
/// contiguously, with no header. The filter's client data carries a recursive
/// type tree — atomic (`[1, size, order, precision, offset]`), array
/// (`[2, total_size, <base>]`) and compound
/// (`[3, total_size, nmembers, (offset, <node>)*]`) — preceded by
/// `[nparms, flag, nelmts]`. Decompression walks the tree once per element,
/// placing each field's bits at its byte/bit offset in a zero-filled element
/// (HDF5's canonical reduced-precision layout). Sign-extension of reduced
/// precision signed integers is the datatype reader's job. Atomic floats are
/// encoded as full-precision atomics and handled transparently.
fn nbit_decompress(data: &[u8], cd: &[u32], expected_bytes: usize) -> Result<Vec<u8>, FormatError> {
    if cd.len() < 4 {
        return Err(FormatError::ChunkedReadError(
            "nbit: missing filter client data".into(),
        ));
    }
    let nelmts = cd[2] as usize;
    let mut idx = 3;
    let root = parse_nbit_node(cd, &mut idx, 0)?;
    let elem_size = root.byte_size();
    if elem_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "nbit: zero element size".into(),
        ));
    }

    let total = nelmts
        .checked_mul(elem_size)
        .ok_or_else(|| FormatError::ChunkedReadError("nbit: size overflow".into()))?;
    // The decoded output must match the chunk's uncompressed size; reject a
    // count that would over-allocate.
    if expected_bytes != 0 && total > expected_bytes {
        return Err(FormatError::ChunkedReadError(
            "nbit: element count exceeds chunk size".into(),
        ));
    }
    let mut out = vec![0u8; total];
    let mut br = BitReader { data, pos: 0 };
    for elem in out.chunks_exact_mut(elem_size) {
        decode_nbit_node(&root, &mut br, elem, 0)?;
    }
    Ok(out)
}

/// Decompress zlib-compressed data.
#[cfg(feature = "deflate")]
fn deflate_decompress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    // Try system zlib first on macOS (Apple's ARM64-optimized libz is ~1.4x
    // faster at decompression than zlib-ng on Apple Silicon).
    #[cfg(all(target_os = "macos", feature = "system-zlib-decompress"))]
    {
        if let Ok(result) = sysz::decompress(data) {
            return Ok(result);
        }
        // Fall through to flate2 on error
    }

    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut result = Vec::new();
    decoder
        .read_to_end(&mut result)
        .map_err(|e| FormatError::DecompressionError(e.to_string()))?;
    Ok(result)
}

/// Direct FFI to Apple's system libz for fast decompression.
///
/// Apple's `/usr/lib/libz.1.dylib` on ARM64 includes hardware-optimized
/// inflate that is ~1.4x faster than zlib-ng for decompression.
/// We use `uncompress` for known-size chunks (the common HDF5 case).
#[cfg(all(target_os = "macos", feature = "system-zlib-decompress"))]
mod sysz {
    use std::os::raw::{c_int, c_ulong};

    // Link against system libz (Apple's optimized build)
    // SAFETY: libz symbols are linked via #[link(name="z")]. Function signatures match zlib.h.
    #[link(name = "z")]
    unsafe extern "C" {
        fn uncompress(
            dest: *mut u8,
            dest_len: *mut c_ulong,
            source: *const u8,
            source_len: c_ulong,
        ) -> c_int;
    }

    const Z_OK: c_int = 0;
    const Z_BUF_ERROR: c_int = -5;

    /// Maximum decompressed output size (256 MiB) to prevent unbounded allocation
    /// from malicious or corrupted compressed data.
    const MAX_DECOMPRESS_SIZE: c_ulong = 256 * 1024 * 1024;

    pub(super) fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        // Start with 8x input as estimate, grow if needed
        let mut out_len = (data.len() * 8) as c_ulong;
        let mut output = vec![0u8; out_len as usize];

        loop {
            let mut actual_len = out_len;
            // SAFETY: zlib_ng FFI function requires valid input/output buffers and
            // proper zlib stream state. All buffer sizes are validated before this call.
            let ret = unsafe {
                uncompress(
                    output.as_mut_ptr(),
                    &mut actual_len,
                    data.as_ptr(),
                    data.len() as c_ulong,
                )
            };

            match ret {
                Z_OK => {
                    output.truncate(actual_len as usize);
                    return Ok(output);
                }
                Z_BUF_ERROR => {
                    // Buffer too small, double it
                    out_len *= 2;
                    if out_len > MAX_DECOMPRESS_SIZE {
                        return Err(format!(
                            "system zlib decompressed output would exceed {} MiB limit",
                            MAX_DECOMPRESS_SIZE / 1024 / 1024
                        ));
                    }
                    output.resize(out_len as usize, 0);
                }
                err => return Err(format!("system zlib uncompress failed: {err}")),
            }
        }
    }
}

#[cfg(not(feature = "deflate"))]
fn deflate_decompress(_data: &[u8]) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Compress data with zlib.
#[cfg(feature = "deflate")]
fn deflate_compress(data: &[u8], level: u32) -> Result<Vec<u8>, FormatError> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    encoder
        .write_all(data)
        .map_err(|e| FormatError::CompressionError(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| FormatError::CompressionError(e.to_string()))
}

#[cfg(not(feature = "deflate"))]
fn deflate_compress(_data: &[u8], _level: u32) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Decompress LZ4 data. Format: 4 bytes LE original size + LZ4 block data.
#[cfg(feature = "lz4")]
fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    if data.len() < 4 {
        return Err(FormatError::DecompressionError(
            "lz4: data too short".into(),
        ));
    }
    let orig_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    lz4_flex::block::decompress(&data[4..], orig_size)
        .map_err(|e| FormatError::DecompressionError(format!("lz4: {e}")))
}

#[cfg(not(feature = "lz4"))]
fn lz4_decompress(_data: &[u8]) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_LZ4))
}

/// Compress data with LZ4 block format. Format: 4 bytes LE original size + LZ4 block data.
#[cfg(feature = "lz4")]
fn lz4_compress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    let compressed = lz4_flex::block::compress(data);
    let mut result = Vec::with_capacity(4 + compressed.len());
    result.extend_from_slice(&(data.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    Ok(result)
}

#[cfg(not(feature = "lz4"))]
fn lz4_compress(_data: &[u8]) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_LZ4))
}

/// Decompress zstd data.
#[cfg(feature = "zstd")]
fn zstd_decompress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    zstd::decode_all(data).map_err(|e| FormatError::DecompressionError(format!("zstd: {e}")))
}

#[cfg(not(feature = "zstd"))]
fn zstd_decompress(_data: &[u8]) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_ZSTD))
}

/// Compress data with zstd.
#[cfg(feature = "zstd")]
fn zstd_compress(data: &[u8], level: u32) -> Result<Vec<u8>, FormatError> {
    zstd::encode_all(data, level as i32)
        .map_err(|e| FormatError::CompressionError(format!("zstd: {e}")))
}

#[cfg(not(feature = "zstd"))]
fn zstd_compress(_data: &[u8], _level: u32) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_ZSTD))
}

/// Unshuffle (decompress direction): reconstruct interleaved element bytes.
/// On disk: all byte-0s of each element together, then all byte-1s, etc.
/// Output: elements in natural order.
fn shuffle_decompress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(FormatError::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    for i in 0..num_elements {
        for j in 0..element_size {
            result[i * element_size + j] = data[j * num_elements + i];
        }
    }

    Ok(result)
}

/// Shuffle (compress direction): group bytes by position within each element.
///
/// This is an AoS→SoA byte transpose.  The hot paths for 4-byte (f32) and
/// 8-byte (f64) elements use unrolled word loads so LLVM can auto-vectorise
/// them into SSE2/AVX2/NEON instructions.  All other element sizes fall through
/// to a cache-blocked scalar loop that avoids the strided-write penalty of the
/// naïve double loop.
fn shuffle_compress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(FormatError::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    match element_size {
        4 => shuffle_compress_4(data, num_elements, &mut result),
        8 => shuffle_compress_general(data, num_elements, element_size, &mut result),
        _ => shuffle_compress_general(data, num_elements, element_size, &mut result),
    }

    Ok(result)
}

/// AoS→SoA for 4-byte elements (f32).
///
/// Processes 4 elements (16 bytes) per iteration using u32 word loads.
/// LLVM vectorises the four parallel shift+mask sequences into SIMD byte
/// deinterleave instructions (e.g., x86 PSHUFB, AArch64 TBL).
#[inline]
fn shuffle_compress_4(data: &[u8], n: usize, result: &mut [u8]) {
    let n4 = n / 4;

    for block in 0..n4 {
        let src = block * 16;
        let w0 = u32::from_le_bytes(data[src..src + 4].try_into().unwrap());
        let w1 = u32::from_le_bytes(data[src + 4..src + 8].try_into().unwrap());
        let w2 = u32::from_le_bytes(data[src + 8..src + 12].try_into().unwrap());
        let w3 = u32::from_le_bytes(data[src + 12..src + 16].try_into().unwrap());

        let o0 = block * 4;
        result[o0] = w0 as u8;
        result[o0 + 1] = w1 as u8;
        result[o0 + 2] = w2 as u8;
        result[o0 + 3] = w3 as u8;

        let o1 = n + block * 4;
        result[o1] = (w0 >> 8) as u8;
        result[o1 + 1] = (w1 >> 8) as u8;
        result[o1 + 2] = (w2 >> 8) as u8;
        result[o1 + 3] = (w3 >> 8) as u8;

        let o2 = 2 * n + block * 4;
        result[o2] = (w0 >> 16) as u8;
        result[o2 + 1] = (w1 >> 16) as u8;
        result[o2 + 2] = (w2 >> 16) as u8;
        result[o2 + 3] = (w3 >> 16) as u8;

        let o3 = 3 * n + block * 4;
        result[o3] = (w0 >> 24) as u8;
        result[o3 + 1] = (w1 >> 24) as u8;
        result[o3 + 2] = (w2 >> 24) as u8;
        result[o3 + 3] = (w3 >> 24) as u8;
    }

    // Remainder (n not a multiple of 4)
    for i in (n4 * 4)..n {
        for j in 0..4usize {
            result[j * n + i] = data[i * 4 + j];
        }
    }
}

/// Cache-blocked AoS→SoA for arbitrary element sizes.
///
/// Processes BLOCK elements at a time so the input tile stays in L1 cache
/// while all `element_size` byte-planes are extracted from it.  This avoids
/// the strided-write cache penalty of the naïve double loop.
#[inline]
fn shuffle_compress_general(data: &[u8], n: usize, element_size: usize, result: &mut [u8]) {
    const BLOCK: usize = 64;
    for block_start in (0..n).step_by(BLOCK) {
        let block_end = (block_start + BLOCK).min(n);
        for j in 0..element_size {
            let out_base = j * n;
            for i in block_start..block_end {
                result[out_base + i] = data[i * element_size + j];
            }
        }
    }
}

/// Compute HDF5 Fletcher32 checksum over data.
/// HDF5 uses a modified Fletcher32 that operates on 16-bit words.
///
/// Optimized with wider accumulators: processes blocks of 360 words before
/// taking the modulo, reducing the number of expensive modulo operations.
/// (360 is the maximum block size that avoids u32 overflow for sum2.)
fn fletcher32_compute(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    // Process in blocks of 360 16-bit words (720 bytes) to delay modulo.
    // Max sum1 before mod: 360 * 65535 = 23_592_600 < u32::MAX
    // Max sum2 before mod: 360 * 23_592_600 ~ 8.5B > u32::MAX, but actual
    // sum2 accumulates incrementally, so worst case is 360*360*65535/2 which
    // fits in u64. We use u32 with block size 360 which is safe.
    const BLOCK_WORDS: usize = 360;
    const BLOCK_BYTES: usize = BLOCK_WORDS * 2;

    let mut offset = 0;
    let len = data.len();

    while offset + BLOCK_BYTES <= len {
        let end = offset + BLOCK_BYTES;
        let mut i = offset;
        while i < end {
            let val = ((data[i] as u32) << 8) | (data[i + 1] as u32);
            sum1 += val;
            sum2 += sum1;
            i += 2;
        }
        sum1 %= 65535;
        sum2 %= 65535;
        offset = end;
    }

    // Handle remaining bytes
    while offset < len {
        let val = if offset + 1 < len {
            ((data[offset] as u32) << 8) | (data[offset + 1] as u32)
        } else {
            (data[offset] as u32) << 8
        };
        sum1 = (sum1 + val) % 65535;
        sum2 = (sum2 + sum1) % 65535;
        offset += 2;
    }

    (sum2 << 16) | sum1
}

/// Verify Fletcher32 checksum and strip it from the data.
/// The last 4 bytes are the stored checksum.
fn fletcher32_verify(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    if data.len() < 4 {
        return Err(FormatError::FilterError(
            "fletcher32: data too short for checksum".into(),
        ));
    }
    let payload = &data[..data.len() - 4];
    let stored = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed = fletcher32_compute(payload);
    if stored != computed {
        return Err(FormatError::Fletcher32Mismatch {
            expected: stored,
            computed,
        });
    }
    Ok(payload.to_vec())
}

/// Append Fletcher32 checksum to data.
fn fletcher32_append(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    let checksum = fletcher32_compute(data);
    let mut result = data.to_vec();
    result.extend_from_slice(&checksum.to_le_bytes());
    Ok(result)
}

// ---------------------------------------------------------------------------
// Pcodec — lossless numerical compression (arXiv:2502.06112)
// ---------------------------------------------------------------------------

#[cfg(feature = "pcodec")]
fn pcodec_compress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    use pco::ChunkConfig;
    use pco::standalone::simple_compress;
    let config = ChunkConfig::default();
    match element_size {
        4 => {
            let nums: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            simple_compress(&nums, &config)
                .map_err(|e| FormatError::CompressionError(format!("pco: {e}")))
        }
        8 => {
            let nums: Vec<f64> = data
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            simple_compress(&nums, &config)
                .map_err(|e| FormatError::CompressionError(format!("pco: {e}")))
        }
        _ => {
            let nums: Vec<u32> = data
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            simple_compress(&nums, &config)
                .map_err(|e| FormatError::CompressionError(format!("pco: {e}")))
        }
    }
}

#[cfg(not(feature = "pcodec"))]
fn pcodec_compress(_data: &[u8], _element_size: usize) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_PCODEC))
}

#[cfg(feature = "pcodec")]
fn pcodec_decompress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    use pco::standalone::simple_decompress;
    match element_size {
        4 => {
            let nums = simple_decompress::<f32>(data)
                .map_err(|e| FormatError::DecompressionError(format!("pco: {e}")))?;
            Ok(nums.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        8 => {
            let nums = simple_decompress::<f64>(data)
                .map_err(|e| FormatError::DecompressionError(format!("pco: {e}")))?;
            Ok(nums.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        _ => {
            let nums = simple_decompress::<u32>(data)
                .map_err(|e| FormatError::DecompressionError(format!("pco: {e}")))?;
            Ok(nums.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
    }
}

#[cfg(not(feature = "pcodec"))]
fn pcodec_decompress(_data: &[u8], _element_size: usize) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_PCODEC))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_pipeline::FilterDescription;

    // --- Deflate tests ---

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = deflate_compress(&data, 6).unwrap();
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_python_zlib() {
        // Data compressed with Python: zlib.compress(bytes(range(10)), 6)
        // python3 -c "import zlib; print(list(zlib.compress(bytes(range(10)), 6)))"
        // = [120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 1, 123, 0, 170]
        let compressed: Vec<u8> = vec![
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_verifiable() {
        // Compress data and verify it decompresses correctly
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let compressed = deflate_compress(&data, 6).unwrap();
        assert!(!compressed.is_empty());
        let decompressed = deflate_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- Shuffle tests ---

    #[test]
    fn shuffle_roundtrip_f64() {
        // 4 f64 values = 32 bytes, element_size=8
        let data: Vec<u8> = (0..32).collect();
        let shuffled = shuffle_compress(&data, 8).unwrap();
        let unshuffled = shuffle_decompress(&shuffled, 8).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_roundtrip_i32() {
        // 8 i32 values = 32 bytes, element_size=4
        let data: Vec<u8> = (0..32).collect();
        let shuffled = shuffle_compress(&data, 4).unwrap();
        let unshuffled = shuffle_decompress(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_known_pattern() {
        // 2 elements of size 4: [A0 A1 A2 A3 B0 B1 B2 B3]
        // After shuffle: [A0 B0 A1 B1 A2 B2 A3 B3]
        let data = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let shuffled = shuffle_compress(&data, 4).unwrap();
        assert_eq!(
            shuffled,
            vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]
        );
    }

    // --- Fletcher32 tests ---

    #[test]
    fn fletcher32_roundtrip() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let with_checksum = fletcher32_append(&data).unwrap();
        assert_eq!(with_checksum.len(), data.len() + 4);
        let verified = fletcher32_verify(&with_checksum).unwrap();
        assert_eq!(verified, data);
    }

    #[test]
    fn fletcher32_known_checksum() {
        // Verify checksum is deterministic
        let data = vec![0u8; 16];
        let with_checksum = fletcher32_append(&data).unwrap();
        let checksum = u32::from_le_bytes([
            with_checksum[16],
            with_checksum[17],
            with_checksum[18],
            with_checksum[19],
        ]);
        // All zeros -> sum1=0, sum2=0 -> checksum=0
        assert_eq!(checksum, 0);

        // Non-zero data
        let data2 = vec![1u8, 0, 0, 0];
        let with_checksum2 = fletcher32_append(&data2).unwrap();
        let verified = fletcher32_verify(&with_checksum2).unwrap();
        assert_eq!(verified, data2);
    }

    #[test]
    fn fletcher32_mismatch_detected() {
        let data = vec![1u8, 2, 3, 4];
        let mut with_checksum = fletcher32_append(&data).unwrap();
        // Corrupt checksum
        let last = with_checksum.len() - 1;
        with_checksum[last] ^= 0xFF;
        let result = fletcher32_verify(&with_checksum);
        assert!(matches!(
            result,
            Err(FormatError::Fletcher32Mismatch { .. })
        ));
    }

    // --- Pipeline tests ---

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_deflate_only() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 1).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 1).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        // 25 f64 values (200 bytes)
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..160).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate_fletcher32() {
        let pipeline = FilterPipeline {
            version: 1,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![9],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        // Use realistic f64-sized data
        let data: Vec<u8> = (0..80).map(|i| (i * 3 % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- LZ4 tests ---

    #[test]
    #[cfg(feature = "lz4")]
    fn lz4_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = lz4_compress(&data).unwrap();
        let decompressed = lz4_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn pipeline_lz4_only() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_LZ4,
                name: Some("lz4".into()),
                flags: 0,
                client_data: vec![],
            }],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 1).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 1).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn pipeline_shuffle_lz4() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_LZ4,
                    name: Some("lz4".into()),
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- Zstd tests ---

    #[test]
    #[cfg(feature = "zstd")]
    fn zstd_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = zstd_compress(&data, 3).unwrap();
        let decompressed = zstd_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn pipeline_zstd_only() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_ZSTD,
                name: Some("zstd".into()),
                flags: 0,
                client_data: vec![3],
            }],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 1).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 1).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn pipeline_shuffle_zstd_fletcher32() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_ZSTD,
                    name: Some("zstd".into()),
                    flags: 0,
                    client_data: vec![1],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..160).map(|i| (i % 256) as u8).collect();
        let compressed = compress_chunk(&data, &pipeline, 8).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, data.len(), 8).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- scale-offset (filter id 6) -------------------------------------------
    // Inputs below are real compressed chunks + client data captured from
    // h5py 3.16 / HDF5 2.0 (`scaleoffset=0`), so they guard the decoder against
    // the reference implementation without needing h5py at test time.

    fn i32_le(vals: &[i32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn scaleoffset_int_basic() {
        // i32 [0,1,2,3], chunk of 4, default fill 0 (element 0 is the fill).
        let cd = [2u32, 0, 4, 0, 4, 1, 0, 1, 0];
        let raw = [
            0x02, 0x00, 0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6, 0x00,
        ];
        assert_eq!(scaleoffset_decompress(&raw, &cd, 0).unwrap(), i32_le(&[0, 1, 2, 3]));
    }

    #[test]
    fn scaleoffset_int_negative() {
        // i32 [-5,-3,-1,0,2,4,7,9], chunk of 8, fill 0 (minval = -5).
        let cd = [2u32, 0, 8, 0, 4, 1, 0, 1, 0];
        let raw = [
            0x04, 0x00, 0x00, 0x00, 0x08, 0xfb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x4f, 0x79, 0xce, 0x00,
        ];
        assert_eq!(
            scaleoffset_decompress(&raw, &cd, 0).unwrap(),
            i32_le(&[-5, -3, -1, 0, 2, 4, 7, 9])
        );
    }

    #[test]
    fn scaleoffset_uint() {
        // u32 [100..109], chunk of 10, unsigned (cd[5]==0), minval 100.
        let cd = [2u32, 0, 10, 0, 4, 0, 0, 1, 0];
        let raw = [
            0x04, 0x00, 0x00, 0x00, 0x08, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0x00,
        ];
        let expected: Vec<u8> = (100u32..110).flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(scaleoffset_decompress(&raw, &cd, 0).unwrap(), expected);
    }

    fn as_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn scaleoffset_float_dscale_d1() {
        // f32 [0,1,2,3], D=1, default fill 0 (element 0 is the fill).
        let cd = [0u32, 1, 4, 1, 4, 0, 0, 1, 0];
        let raw = [
            0x05, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x15, 0x40,
        ];
        let got = as_f32(&scaleoffset_decompress(&raw, &cd, 0).unwrap());
        assert_eq!(got, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn scaleoffset_float_dscale_d3() {
        // f32 [0,0.1,0.2,0.3], D=3 (lossy reconstruction within 10^-3).
        let cd = [0u32, 3, 4, 1, 4, 0, 0, 1, 0];
        let raw = [
            0x08, 0x00, 0x00, 0x00, 0x08, 0xcd, 0xcc, 0xcc, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x64, 0xc8, 0x00,
        ];
        let got = as_f32(&scaleoffset_decompress(&raw, &cd, 0).unwrap());
        let exp = [0.0f32, 0.1, 0.2, 0.3];
        assert_eq!(got.len(), 4);
        for (g, e) in got.iter().zip(exp.iter()) {
            assert!((g - e).abs() < 1e-3, "got {g} expected {e}");
        }
    }

    fn as_f64(bytes: &[u8]) -> Vec<f64> {
        bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn scaleoffset_float_escale_e1() {
        // f64 [0.0, 2.0, 4.0, 6.0], E=1 (×2^1=2), fill_defined=0.
        // cd: scale_type=1, E=1, nelmts=4, elem_size=8.
        let cd = [1u32, 1, 4, 0, 8, 0, 0, 0];
        let raw: &[u8] = &[
            2, 0, 0, 0, // minbits=2
            8,          // minval_width=8
            0, 0, 0, 0, 0, 0, 0, 0, // minval=0.0f64
            0, 0, 0, 0, 0, 0, 0, 0, // 8 reserved bytes
            0x1B, // packed codes: 00 01 10 11 MSB-first
        ];
        let got = as_f64(&scaleoffset_decompress(raw, &cd, 0).unwrap());
        assert_eq!(got, vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn scaleoffset_float_escale_neg_exp() {
        // f64 [0.0, 0.5, 1.0, 1.5], E=-1 (×2^-1=0.5), fill_defined=0.
        // cd[1] = 0xFFFF_FFFF which casts to i32 = -1.
        let cd = [1u32, 0xFFFF_FFFF, 4, 0, 8, 0, 0, 0];
        let raw: &[u8] = &[
            2, 0, 0, 0, // minbits=2
            8,          // minval_width=8
            0, 0, 0, 0, 0, 0, 0, 0, // minval=0.0f64
            0, 0, 0, 0, 0, 0, 0, 0, // 8 reserved bytes
            0x1B, // packed codes: 00 01 10 11 MSB-first
        ];
        let got = as_f64(&scaleoffset_decompress(raw, &cd, 0).unwrap());
        let exp = [0.0f64, 0.5, 1.0, 1.5];
        for (g, e) in got.iter().zip(exp.iter()) {
            assert!((g - e).abs() < 1e-9, "got {g} expected {e}");
        }
    }

    // --- N-Bit (filter id 5) --------------------------------------------------
    // Real compressed chunks + client data from h5py 3.16 / HDF5 2.0. The
    // expected outputs are the canonical zero-filled element bytes (verified
    // equal to the contiguous, un-filtered on-disk layout of the same type).

    #[test]
    fn nbit_unsigned_12bit() {
        // u32 storage, 12-bit precision: [0, 1, 4095, 2048].
        let cd = [8u32, 0, 4, 1, 4, 0, 12, 0];
        let raw = [0x00, 0x00, 0x01, 0xff, 0xf8, 0x00, 0x00];
        let expected: Vec<u8> = [0u32, 1, 4095, 2048]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(nbit_decompress(&raw, &cd, 0).unwrap(), expected);
    }

    #[test]
    fn nbit_signed_16bit_zero_filled() {
        // i32 storage, 16-bit precision: [-1, -50, 100, -1000]. N-Bit restores
        // the canonical zero-filled layout (high 16 bits zero); the datatype
        // reader is responsible for sign-extending reduced-precision integers.
        let cd = [8u32, 0, 4, 1, 4, 0, 16, 0];
        let raw = [0xff, 0xff, 0xff, 0xce, 0x00, 0x64, 0xfc, 0x18, 0x00];
        // Canonical bytes captured from an equivalent un-filtered dataset.
        let expected: Vec<u8> = vec![
            0xff, 0xff, 0x00, 0x00, // 0x0000ffff
            0xce, 0xff, 0x00, 0x00, // 0x0000ffce
            0x64, 0x00, 0x00, 0x00, // 0x00000064
            0x18, 0xfc, 0x00, 0x00, // 0x0000fc18
        ];
        assert_eq!(nbit_decompress(&raw, &cd, 0).unwrap(), expected);
    }

    #[test]
    fn nbit_compound_int_members() {
        // Compound { a: i32@0 prec 16, b: u32@4 prec 8 }, 3 elements.
        // data = [(-1,200),(1000,7),(-32768,255)]. Captured from HDF5 2.0.
        let cd = [18u32, 0, 3, 3, 8, 2, 0, 1, 4, 0, 16, 0, 4, 1, 4, 0, 8, 0];
        let raw = [0xff, 0xff, 0xc8, 0x03, 0xe8, 0x07, 0x80, 0x00, 0xff, 0x00];
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            0xff,0xff,0x00,0x00, 0xc8,0x00,0x00,0x00, // (-1, 200)
            0xe8,0x03,0x00,0x00, 0x07,0x00,0x00,0x00, // (1000, 7)
            0x00,0x80,0x00,0x00, 0xff,0x00,0x00,0x00, // (-32768, 255)
        ];
        assert_eq!(nbit_decompress(&raw, &cd, 0).unwrap(), expected);
    }

    #[test]
    fn nbit_compound_with_array_member() {
        // Compound { a: array(2,) of i32 prec 16 @0; b: u32@8 prec 8 }, 2 elements.
        // data = [([-1,100],200), ([1000,-32768],7)].
        let cd = [20u32, 0, 2, 3, 12, 2, 0, 2, 8, 1, 4, 0, 16, 0, 8, 1, 4, 0, 8, 0];
        let raw = [0xff, 0xff, 0x00, 0x64, 0xc8, 0x03, 0xe8, 0x80, 0x00, 0x07, 0x00];
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            0xff,0xff,0x00,0x00, 0x64,0x00,0x00,0x00, 0xc8,0x00,0x00,0x00, // ([-1,100], 200)
            0xe8,0x03,0x00,0x00, 0x00,0x80,0x00,0x00, 0x07,0x00,0x00,0x00, // ([1000,-32768], 7)
        ];
        assert_eq!(nbit_decompress(&raw, &cd, 0).unwrap(), expected);
    }

    #[test]
    fn nbit_compound_with_float_member() {
        // Compound { i: i32@0 prec 16; f: f32@4 prec 32 }, 2 elements.
        // data = [(-1, 1.5), (100, -2.5)]. The float member is a full-precision
        // atomic; its bits are packed MSB-first and restored to LE storage.
        let cd = [18u32, 0, 2, 3, 8, 2, 0, 1, 4, 0, 16, 0, 4, 1, 4, 0, 32, 0];
        let raw = [
            0xff, 0xff, 0x3f, 0xc0, 0x00, 0x00, 0x00, 0x64, 0xc0, 0x20, 0x00, 0x00, 0x00,
        ];
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            0xff,0xff,0x00,0x00, 0x00,0x00,0xc0,0x3f, // (-1, 1.5)
            0x64,0x00,0x00,0x00, 0x00,0x00,0x20,0xc0, // (100, -2.5)
        ];
        assert_eq!(nbit_decompress(&raw, &cd, 0).unwrap(), expected);
    }

    // ----- Adversarial / hardening: malformed filter data must not panic -----

    #[test]
    fn scaleoffset_minbits_64_does_not_panic() {
        // minbits == 64 previously overflowed `1u64 << minbits` computing the
        // fill code. cd: int, nelmts=1, elem_size=8, fill_defined=1.
        let cd = [2u32, 0, 1, 0, 8, 1, 0, 1];
        let mut data = Vec::new();
        data.extend_from_slice(&64u32.to_le_bytes()); // minbits = 64
        data.push(8); // minval_width
        data.extend_from_slice(&0i64.to_le_bytes()); // minval
        data.extend_from_slice(&[0u8; 8]); // reserved
        data.extend_from_slice(&[0u8; 8]); // one 64-bit code
        // Must return a Result (Ok or Err) without panicking.
        let _ = scaleoffset_decompress(&data, &cd, 8);
    }

    #[test]
    fn scaleoffset_huge_nelmts_bounded_by_chunk_size() {
        // minbits == 0 (no packed payload) with a giant nelmts must not try to
        // allocate when the expected chunk size is small.
        let cd = [2u32, 0, u32::MAX, 0, 4, 0, 0, 0];
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes()); // minbits = 0
        data.push(4);
        data.extend_from_slice(&[0u8; 4]);
        assert!(scaleoffset_decompress(&data, &cd, 64).is_err());
    }

    #[test]
    fn nbit_deeply_nested_array_is_rejected() {
        // A chain of ARRAY nodes (class 2) far deeper than NBIT_MAX_DEPTH must
        // error rather than recurse into a stack overflow. Layout per level:
        // [class=2, total]. Terminate with a (never-reached) atomic.
        let mut cd = vec![0u32, 0, 1]; // nparms, flag, nelmts
        for _ in 0..500 {
            cd.push(2); // ARRAY
            cd.push(8); // total size
        }
        cd.extend_from_slice(&[1, 8, 0, 8, 0]); // atomic leaf
        assert!(nbit_decompress(&[], &cd, 0).is_err());
    }

    #[test]
    fn nbit_atomic_bit_offset_overflow_is_rejected() {
        // bit_offset + precision near u32::MAX must not overflow the check.
        let cd = [0u32, 0, 1, 1, 8, 0, u32::MAX, u32::MAX];
        assert!(nbit_decompress(&[0u8; 8], &cd, 0).is_err());
    }

    #[test]
    fn nbit_huge_nelmts_bounded_by_chunk_size() {
        // Valid 1-byte atomic but an enormous element count; the expected chunk
        // size bounds the allocation.
        let cd = [0u32, 0, u32::MAX, 1, 1, 0, 8, 0];
        assert!(nbit_decompress(&[0u8; 4], &cd, 16).is_err());
    }

    #[test]
    fn scaleoffset_truncated_inputs_do_not_panic() {
        assert!(scaleoffset_decompress(&[], &[2, 0, 1, 0, 4, 0, 0, 0], 4).is_err());
        assert!(scaleoffset_decompress(&[0u8; 3], &[2, 0, 1, 0, 4, 0, 0, 0], 4).is_err());
        // Missing client data entirely.
        assert!(scaleoffset_decompress(&[0u8; 32], &[2, 0], 4).is_err());
    }
}
