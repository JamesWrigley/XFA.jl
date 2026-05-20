module XFA

using XfaEngine: XfaEngine

const Maybe{T} = Union{T, Nothing}

struct Point2d
    x::Float64
    y::Float64
end

include("util.jl")
include("binning.jl")
include("gui.jl")
include("settings.jl")

# using PrecompileTools: @compile_workload

end
