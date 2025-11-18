using Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
Pkg.instantiate()

using Printf
using Serialization
using BlockArrays
using RobustBeliefGame

const senate_module_path = joinpath(@__DIR__, "Senate.jl")
if !isdefined(Main, :Senate)
    @eval Main begin
        include($senate_module_path)
    end
end
const Senate = Main.Senate

# Register UUID for deserialization at module load time
const senate_uuid = Base.UUID("a2515029-de12-424a-9371-454911e3b6f1")
const senate_pkgid = Base.PkgId(senate_uuid, "Senate")
if !haskey(Base.loaded_modules, senate_pkgid)
    Base.loaded_modules[senate_pkgid] = Senate
end

const PARAM_KEYS = ["p1t", "p2t", "p2nm", "dmt", "p2bpdss", "p1ow", "h", "gtis"]

mutable struct FileParams
    p1t::String
    p2t::String
    p2nm::Float64
    dmt::String
    p2bpdss::Float64
    p1ow::Vector{Float64}
    h::Int
    gtis::Vector{Float64}
    original_path::String

    FileParams() = new("", "", 0.0, "", 0.0, [], 0, [], "")
end

function Base.show(io::IO, p::FileParams)
    for field in fieldnames(FileParams)
        println(io, "$field: $(getfield(p, field))")
    end
end

function parse_filename(filepath::String)
    filename = basename(filepath)
    params = FileParams()
    params.original_path = filepath

    function get_val(key)
        regex = Regex(key * "_([^_]+(?:_[^_]+)?)")
        m = match(regex, filename)
        return m === nothing ? "" : m.captures[1]
    end

    params.p1t = get_val("p1t")
    params.p2t = get_val("p2t")
    params.dmt = get_val("dmt")

    p2nm_m = match(r"p2nm_([\d\.]+)", filename)
    if p2nm_m !== nothing
        params.p2nm = parse(Float64, p2nm_m.captures[1])
    end

    p2bpdss_matches = collect(eachmatch(r"p2bpdss_(?!p2bpdss)([\d\.]+)", filename))
    if !isempty(p2bpdss_matches)
        # Use the last match to handle cases where p2bpdss appears multiple times
        p2bpdss_m = p2bpdss_matches[end]
        params.p2bpdss = parse(Float64, p2bpdss_m.captures[1])
    end

    h_m = match(r"h_(\d+)", filename)
    if h_m !== nothing
        params.h = parse(Int, h_m.captures[1])
    end

    gtis_match = match(r"gtis_(\[.*?\])", filename)
    if gtis_match !== nothing
        gtis_str = gtis_match.captures[1]
        gtis_str = replace(gtis_str, "[" => "", "]" => "")
        params.gtis = [parse(Float64, s) for s in split(gtis_str, ", ")]
    end
    
    # Parse p1ow (p1_obstacle_weights) - format is p1ow_[value] where value is a float
    # Handle both p1ow_[0.1] and p1ow_[0.1]_next_param formats
    p1ow_match = match(r"p1ow_(\[.*?\])", filename)
    if p1ow_match !== nothing
        p1ow_str = p1ow_match.captures[1]
        p1ow_str = replace(p1ow_str, "[" => "", "]" => "")
        # Handle comma-separated values (though typically just one value)
        p1ow_str = replace(p1ow_str, ", " => ",")
        if !isempty(p1ow_str)
            params.p1ow = [parse(Float64, s) for s in split(p1ow_str, ",") if !isempty(s)]
        else
            params.p1ow = []
        end
    else
        # If not found, set to empty vector
        params.p1ow = []
    end
    
    return params
end

function combine_files(output_path, source_paths)
    if isempty(source_paths)
        return
    end

    combined_results = []
    for source_path in source_paths
        try
            open(source_path, "r") do in_file
                data = deserialize(in_file)
                
                # Parse filename to extract experiment name and parameters
                file_params = parse_filename(source_path)
                filename_without_ext = Base.splitext(basename(source_path))[1]
                
                # Convert FileParams to a Dict for the params field
                params_dict = Dict{Symbol, Any}()
                for key in fieldnames(FileParams)
                    if key != :original_path
                        params_dict[key] = getfield(file_params, key)
                    end
                end
                
                # Convert to the format expected by load_solution
                # Format: (params=Dict, fixed=Dict, results=Dict{String,Any}, name=String)
                if data isa Dict{String, Any}
                    # This is a Dict from outputs/runs - convert to NamedTuple format
                    # Handle old format with representative_games (3-tuple) or new format (2-tuple)
                    cleaned_results = Dict{String, Any}()
                    for (trial_id, trial_data) in data
                        try
                            if trial_data isa Tuple && length(trial_data) == 3
                                # Old format: (solutions, games, params) - strip out games
                                # Access games with _ to skip it, in case it contains non-deserializable closures
                                solutions = trial_data[1]
                                params = trial_data[3]
                                cleaned_results[trial_id] = (solutions, params)
                            elseif trial_data isa Tuple && length(trial_data) == 2
                                # New format: (solutions, params)
                                cleaned_results[trial_id] = trial_data
                            else
                                # Unknown format, keep as-is
                                cleaned_results[trial_id] = trial_data
                            end
                        catch inner_e
                            # If accessing tuple elements fails (e.g., due to closure deserialization issues),
                            # try to extract what we can
                            if trial_data isa Tuple && length(trial_data) >= 1
                                println("Warning: Error accessing tuple elements in $trial_id, attempting to extract solutions only: $inner_e")
                                try
                                    solutions = trial_data[1]
                                    # Try to get params if available
                                    if length(trial_data) >= 3
                                        params = trial_data[3]
                                        cleaned_results[trial_id] = (solutions, params)
                                    else
                                        # Skip this trial if we can't get params
                                        println("Skipping trial $trial_id: cannot extract required data")
                                    end
                                catch
                                    println("Skipping trial $trial_id: cannot extract any data")
                                end
                            else
                                println("Skipping trial $trial_id: unexpected format")
                            end
                        end
                    end
                    result_entry = (
                        params=params_dict,
                        fixed=Dict{Symbol, Any}(),
                        results=cleaned_results,
                        name=filename_without_ext
                    )
                    push!(combined_results, result_entry)
                elseif data isa Vector
                    # Already in the correct format (from merged files)
                    append!(combined_results, data)
                else
                    # Unknown format, try to wrap it
                    result_entry = (
                        params=params_dict,
                        fixed=Dict{Symbol, Any}(),
                        results=data,
                        name=filename_without_ext
                    )
                    push!(combined_results, result_entry)
                end
            end
        catch e
            # Check if this is a deserialization error related to closures
            if isa(e, UndefVarError) || (isa(e, ErrorException) && occursin("#", string(e)))
                println("Error deserializing file $source_path (likely due to non-serializable closures in representative_games): $e")
                println("Attempting to skip this file. Consider re-running the experiment to generate a new file without representative_games.")
            else
                println("Error processing file $source_path: $e")
                println("Stacktrace:")
                for (exc, bt) in Base.catch_stack()
                    showerror(stdout, exc, bt)
                    println()
                end
            end
        end
    end

    if !isempty(combined_results)
        open(output_path, "w") do out_file
            serialize(out_file, combined_results)
        end
    end
end

function abbrev_from_key(k::AbstractString)
    join(first.(split(k, "_")))
end

function abbrev_key(k::AbstractString)
    for prefix in ("p1", "p2")
        if startswith(k, prefix)
            rest = k[length(prefix)+2:end]
            return prefix * abbrev_from_key(rest)
        end
    end
    return abbrev_from_key(k)
end

function extract_prefix(filename::String)
    # Extract prefix from filename (asym, cov_asym, or obst_asym)
    if startswith(filename, "cov_asym")
        return "cov_asym"
    elseif startswith(filename, "obst_asym")
        return "obst_asym"
    elseif startswith(filename, "asym")
        return "asym"
    else
        return "asym"  # default fallback
    end
end

function params_to_filename(file_params::FileParams, varying_keys::Vector{Symbol}=Symbol[], prefix::String="asym")
    parts = [prefix]
    
    # Create a dictionary of parameters from the FileParams struct
    combo = Dict{Symbol, Any}()
    for key in fieldnames(FileParams)
        if key != :original_path
            combo[key] = getfield(file_params, key)
        end
    end
    
    # Filter out varying keys
    for key in varying_keys
        delete!(combo, key)
    end
    
    # Build the filename using the abbreviation logic
    for (key, value) in combo
        kstr = String(key)
        
        # Translate from my struct keys to original experiment keys if needed
        # This mapping ensures the abbreviations match the original experiment.
        original_key_map = Dict(
            :p1t => "p1_type",
            :p2t => "p2_type",
            :p2nm => "p2_nature_multiplier",
            :dmt => "dynamics_model_template",
            :p2bpdss => "p2_believes_p1_drift_sensor_scale",
            :p1ow => "p1_obstacle_weights",
            :gtis => "ground_truth_initial_states"
        )
        
        original_kstr = get(original_key_map, key, kstr)
        abbr = abbrev_key(original_kstr)

        # Format value correctly
        val_str = ""
        if value isa Vector
            val_str = "[" * join(value, ",") * "]"
        else
            val_str = replace(string(value), " " => "_")
            # Safety check: if the value string contains the abbreviation, something is wrong
            # This can happen if parsing went wrong and stored "p2bpdss_0.0" instead of "0.0"
            if occursin(abbr, val_str) && abbr != val_str
                @warn "Value string '$val_str' contains abbreviation '$abbr' for key $key. This suggests a parsing error."
                # Try to extract just the numeric part
                num_match = match(r"([\d\.]+)$", val_str)
                if num_match !== nothing
                    val_str = num_match.captures[1]
                end
            end
        end
        
        push!(parts, "$(abbr)_$(val_str)")
    end
    
    return join(parts, "_") * ".dat"
end


function organize_outputs()
    # Define paths
    root_dir = joinpath(@__DIR__, "..")
    runs_dir = joinpath(root_dir, "outputs", "runs")
    organized_dir = joinpath(root_dir, "outputs", "organized")

    # Create directories
    mkpath(organized_dir)

    # List and parse files, grouping by prefix
    all_files_by_prefix = Dict{String, Vector{FileParams}}()
    for f in readdir(runs_dir)
        if endswith(f, ".dat")
            filepath = joinpath(runs_dir, f)
            try
                prefix = extract_prefix(f)
                if !haskey(all_files_by_prefix, prefix)
                    all_files_by_prefix[prefix] = []
                end
                file_params = parse_filename(filepath)
                push!(all_files_by_prefix[prefix], file_params)
            catch e
                println("Skipping file due to parsing error: $f")
                println(e)
            end
        end
    end
    
    total_files = sum(length(files) for files in values(all_files_by_prefix))
    println("Parsed $total_files files across $(length(all_files_by_prefix)) prefix(es).")

    # Process each prefix separately
    for (prefix, all_files) in all_files_by_prefix
        println("\nProcessing prefix: $prefix")
        if prefix != "obst_asym" continue end
        
        # Create prefix-specific directories
        prefix_organized_dir = joinpath(organized_dir, prefix)
        mkpath(joinpath(prefix_organized_dir, "per_param"))
        mkpath(joinpath(prefix_organized_dir, "robustness"))
        mkpath(joinpath(prefix_organized_dir, "p2_wrong_tables"))

        # Grouping 1: per_param
        println("  Grouping by individual parameters...")
        per_param_dir = joinpath(prefix_organized_dir, "per_param")
        param_symbols = [f for f in fieldnames(FileParams) if f != :original_path]

        for param_key in param_symbols
            param_dir = joinpath(per_param_dir, String(param_key))
            mkpath(param_dir)

            other_params = filter(p -> p != param_key && p != :original_path, fieldnames(FileParams))

            groups = Dict()
            for file_params in all_files
                key = Tuple(getfield(file_params, p) for p in other_params)
                if !haskey(groups, key)
                    groups[key] = []
                end
                push!(groups[key], file_params)
            end

            for (key, file_group) in groups
                if isempty(file_group) continue end
                filename = params_to_filename(file_group[1], [param_key], prefix)
                output_path = joinpath(param_dir, filename)
                
                source_paths = [p.original_path for p in file_group]
                combine_files(output_path, source_paths)
            end
        end

        # Grouping 2: robustness
        println("  Grouping by robustness...")
        robustness_dir = joinpath(prefix_organized_dir, "robustness")
        robustness_varying_keys = [:p1t, :p2t]
        robustness_constant_keys = filter(p -> !(p in robustness_varying_keys) && p != :original_path, fieldnames(FileParams))
        
        robustness_groups = Dict()
        for file_params in all_files
            key = Tuple(getfield(file_params, p) for p in robustness_constant_keys)
            if !haskey(robustness_groups, key)
                robustness_groups[key] = []
            end
            push!(robustness_groups[key], file_params)
        end

        for (key, file_group) in robustness_groups
            if isempty(file_group) continue end
            filename = params_to_filename(file_group[1], robustness_varying_keys, prefix)
            output_path = joinpath(robustness_dir, filename)
            
            source_paths = [p.original_path for p in file_group]
            combine_files(output_path, source_paths)
        end

        # Grouping 3: p2_wrong_tables
        println("  Grouping for p2_wrong_tables...")
        p2_wrong_tables_dir = joinpath(prefix_organized_dir, "p2_wrong_tables")
        p2_tables_varying_keys = [:p1t, :p2t, :p2bpdss]
        p2_tables_constant_keys = filter(p -> !(p in p2_tables_varying_keys) && p != :original_path, fieldnames(FileParams))

        p2_tables_groups = Dict()
        for file_params in all_files
            key = Tuple(getfield(file_params, p) for p in p2_tables_constant_keys)
            if !haskey(p2_tables_groups, key)
                p2_tables_groups[key] = []
            end
            push!(p2_tables_groups[key], file_params)
        end

        for (key, file_group) in p2_tables_groups
            if isempty(file_group) continue end
            filename = params_to_filename(file_group[1], p2_tables_varying_keys, prefix)
            output_path = joinpath(p2_wrong_tables_dir, filename)

            source_paths = [p.original_path for p in file_group]
            combine_files(output_path, source_paths)
        end
    end

    println("\nDone.")
end
