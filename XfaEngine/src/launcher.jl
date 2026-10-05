import Pkg
using DistributedNext

include("launcher_utils.jl")


redirect_io()
initialize_logger()


# Throw InterruptExceptions for SIGINT (Ctrl + C) instead of immediately exiting
Base.exit_on_sigint(false)

# Add workers
@info "Engine starting up 🌅 Adding workers..."
# addprocs(2)
@info "Added $(extra_workers()) workers 💪"

try
    # Redirect their IO
    # workers() is [1] if none were added, which is already set up
    if extra_workers() > 0
        @everywhere workers() include("launcher_utils.jl")
        @everywhere workers() redirect_io()
    end

    @everywhere import Revise
    @everywhere import XfaEngine as Engine

    Engine.main()
catch ex
    @error "Caught error, cleaning up workers and exiting" exception=ex
    throw(ex)
finally
    @info "Shutting down workers..."
    rmprocs(workers())
    @info "All workers shutdown, goodbye 👋"
end
