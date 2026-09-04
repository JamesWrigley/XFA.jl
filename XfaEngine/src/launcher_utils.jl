import Dates
import Logging
import InteractiveUtils: versioninfo

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

# The default ConsoleLogger metadata format with a timestamp prepended
function timestamped_metafmt(level, _module, group, id, file, line)
    color, prefix, suffix = Logging.default_metafmt(level, _module, group, id, file, line)
    ts = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    return color, "$(ts) $(prefix)", suffix
end

# Create a customized global logger
function initialize_logger(min_level=Logging.Info)
    # Create a logger that rotates the log files every month and writes in the
    # default ConsoleLogger format.
    file_logger = DatetimeRotatingFileLogger(pwd(), raw"\x\f\a-\e\n\g\i\n\e-yyyy-mm.\l\o\g"; rotation_callback) do io, log
        console = Logging.ConsoleLogger(io, Logging.BelowMinLevel; meta_formatter=timestamped_metafmt)
        Logging.handle_message(console, log.level, log.message, log._module, log.group,
                               log.id, log.file, log.line; log.kwargs...)
    end
    # Start the file over once it grows past 100MB. The seek is needed because
    # truncate() doesn't reset the stream position.
    logger = TransformerLogger(file_logger) do log
        stream = file_logger.logger.stream
        if position(stream) > 100_000_000
            truncate(stream, 0)
            seekstart(stream)
        end

        log
    end
    # Filter by logging level
    logger = MinLevelLogger(logger, min_level)

    Logging.global_logger(logger)
end

"""Return the number of workers added with addprocs()."""
extra_workers() = count(!=(1), workers())
