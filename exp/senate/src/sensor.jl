function base_sensor_model(x::BlockVector, ns::BlockVector; config::PlayerConfig)
    BlockVector(x + config.sensor_noise_scale * ns, config.sensor_dims_per_activist)
end

function drift_sensor_model(x::BlockVector, ns::BlockVector; config::PlayerConfig, undrifted_sensor_model::Function)
    BlockVector(
        undrifted_sensor_model(x, ns; config=config) +
            config.sensor_drift_scale * ones(length(x)),
        config.sensor_dims_per_activist)
end