import Dates
import Logging
import InteractiveUtils: versioninfo

import LoggingFormats: LogFmt
import LoggingExtras: TransformerLogger, DatetimeRotatingFileLogger, MinLevelLogger


"""Redirect stdout and stderr to files based on the worker ID"""
function redirect_io()
    log_name = "worker-$(myid())-stdio.log"
    Threads.@spawn :interactive redirect_stdio(stdout=log_name, stderr=log_name) do
        while true
            sleep(10)
            flush(stdout)
            flush(stderr)
        end
    end

    # Sleep for a bit to wait for the task to launch and the redirect to kick in
    sleep(0.5)

    dt = Dates.format(Dates.now(), "HH:MM:SS on yyyy-mm-dd")
    info_str = "Starting at $(dt) on $(gethostname()) with PID $(getpid())"
    marker = repeat("-", length(info_str))
    println("""

            $(marker)
            $(info_str)
            $(marker)
            """)

    println()
    versioninfo()
    println()
    println()
    flush(stdout)
end

"""Helper function for initialize_logger() to delete old log files"""
function rotation_callback(old_log)
    prefix = old_log[1:length(old_log) - length("-yyyy-mm.log")]
    files = [log for log in readdir(dirname(old_log); join=true)
                 if startswith(log, prefix) && endswith(log, ".log")]

    # Delete the third oldest log file, such that we always store the last two
    # months of logs.
    if length(files) > 2
        rm(files[1])
    end
end

const formatter = LogFmt()

# Create a customized global logger
function initialize_logger(min_level=Logging.Info)
    # Create a logger that rotates the log files every month and uses the logfmt
    # format.
    logger = DatetimeRotatingFileLogger(pwd(), raw"\x\f\a-\e\n\g\i\n\e-yyyy-mm.\l\o\g"; rotation_callback) do io, log
        ts = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
        print(io, "time=", ts, " ")
        formatter(io, log)
    end
    # Filter by logging level
    logger = MinLevelLogger(logger, min_level)

    Logging.global_logger(logger)
end

"""Return the number of workers added with addprocs()."""
extra_workers() = count(!=(1), workers())
