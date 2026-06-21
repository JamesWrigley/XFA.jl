module XFA

using XfaEngine: XfaEngine

const Maybe{T} = Union{T, Nothing}

struct Point2d
    x::Float64
    y::Float64
end

include("fits.jl")
include("util.jl")
include("gui.jl")
include("settings.jl")

# using PrecompileTools: @compile_workload

end
