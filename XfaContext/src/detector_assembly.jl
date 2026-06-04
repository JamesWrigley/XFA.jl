# Holds the single-frame lookup table and the corresponding output image
# size. The LUT is meant to be obtained from extra-geom.
struct AssemblerLUT
    lut::Vector{UInt64}
    frame_size::Tuple{Int, Int}
end

# Build an assembler by shelling out to generate_assembly_lut.py. `geom_file`
# may be omitted for the ePix100 or a single-module JUNGFRAU.
function AssemblerLUT(detector::AbstractString="AGIPD_1MGeometry";
                      geom_file=nothing, n_modules=1, python="python3")
    script = joinpath(@__DIR__, "generate_assembly_lut.py")
    args = ["--detector", detector, "--n_modules", string(n_modules)]
    if !isnothing(geom_file)
        push!(args, expanduser(geom_file))
    end
    bytes = read(`$python $script $args`)
    nx, ny = reinterpret(Int64, @view bytes[1:16])
    lut = collect(reinterpret(UInt64, @view bytes[17:end]))
    return AssemblerLUT(lut, (Int(nx), Int(ny)))
end

nframes(asm::AssemblerLUT, frames::AbstractArray) = length(frames) ÷ length(asm.lut)

output_size(asm::AssemblerLUT, n) = n == 1 ? asm.frame_size : (asm.frame_size..., n)

allocate_output(asm::AssemblerLUT, n=1, ::Type{T}=Float32) where {T} =
    Array{T <: AbstractFloat ? T : Float32}(undef, output_size(asm, n))

# Scatter one flat chunk of module data (frame index trailing) into out_flat
# via the LUT, starting at `offset` in the LUT.
function scatter!(out_flat, lut, chunk_flat, offset, n)
    for f in 1:n
        for i in axes(chunk_flat, 1)
            out_flat[lut[offset + i], f] = chunk_flat[i, f]
        end
    end
end

# Prepare out for assembly: validate its size against n frames and fill with NaN.
function prepare_output!(out::AbstractArray{<:AbstractFloat}, asm::AssemblerLUT, n)
    if length(out) != prod(asm.frame_size) * n
        throw(DimensionMismatch("output array is $(size(out)), expected $(output_size(asm, n))"))
    end
    fill!(out, eltype(out)(NaN))
    return reshape(out, :, n)
end

# Assemble one or more frames into `out`. `frames` holds whole-detector module
# data with the frame index as the trailing axis, in the flat ordering the LUT
# was built against; out is indexed [ny, nx] for a single frame or [ny, nx,
# frame].
function assemble!(out::AbstractArray{<:AbstractFloat}, asm::AssemblerLUT, frames::AbstractArray)
    n = nframes(asm, frames)
    out_flat = prepare_output!(out, asm, n)
    scatter!(out_flat, asm.lut, reshape(frames, :, n), 0, n)
    return out
end

# Same as above but from per-module arrays, in the order the LUT was built
# against, each with the frame index as its trailing axis. The assembly is done
# module-by-module.
function assemble!(out::AbstractArray{<:AbstractFloat}, asm::AssemblerLUT, modules::AbstractVector{<:AbstractArray})
    n = sum(length, modules) ÷ length(asm.lut)
    out_flat = prepare_output!(out, asm, n)

    offset = 0
    for mod in modules
        mod_flat = reshape(mod, :, n)
        scatter!(out_flat, asm.lut, mod_flat, offset, n)
        offset += size(mod_flat, 1)
    end

    return out
end

function assemble(asm::AssemblerLUT, frames::AbstractArray)
    out = allocate_output(asm, nframes(asm, frames), eltype(frames))
    return assemble!(out, asm, frames)
end

function assemble(asm::AssemblerLUT, modules::AbstractVector{<:AbstractArray})
    n = sum(length, modules) ÷ length(asm.lut)
    out = allocate_output(asm, n, eltype(first(modules)))
    return assemble!(out, asm, modules)
end
