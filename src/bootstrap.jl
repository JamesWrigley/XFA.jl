import TOML
using Printf: @sprintf

environment = dirname(Base.active_project())
working_dir = ENV["XFA_WORKING_DIR"]
julia_binary = joinpath(Sys.BINDIR, "julia")

@info "Bootstrapping..." environment working_dir julia_binary

# Check if a worker file already exists
toml_path = joinpath(working_dir, "worker-info.toml")
if isfile(toml_path)
    worker_info = TOML.parsefile(toml_path)
    headnode_pid = worker_info["1"]["pid"]

    # If it does but its old and the process does not exist anymore, delete it
    pid_alive = @ccall(kill(headnode_pid::Cint, 0::Cint)::Cint) == 0
    if !pid_alive
        rm(toml_path)
    end
end

# Launch the engine if necessary
if !isfile(toml_path)
    import XfaEngine
    engine_dir = pkgdir(XfaEngine)
    launcher_script = joinpath(engine_dir, "src", "launcher.jl")
    mkpath(working_dir)
    nthreads = max(10, cld(Sys.CPU_THREADS, 8))
    cd(working_dir) do
        cmd = `$(julia_binary) --project="$(environment)" --color=no --startup-file=no -t $(nthreads),4 $(launcher_script)`
        println("Launching: " * string(cmd))
        run(detach(cmd); wait=false)
    end
end

# Wait for up to 60s for it to start
start = time()
while !isfile(toml_path) || filesize(toml_path) == 0
    elapsed = time() - start
    if elapsed > 60
        error("Timeout while waiting for engine to start in $(working_dir)")
    else
        elapsed_str = @sprintf "%.2fs" elapsed
        println("Waiting for engine to start... $(elapsed_str)")
        sleep(2)
    end
end

# Print the config
println(">>>")
print(read(toml_path, String))
println("<<<")
