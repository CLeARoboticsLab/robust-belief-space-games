function base_sensor_model(x::BlockVector, ns::BlockVector; config::PlayerConfig)
    BlockVector(x + config.sensor_noise_scale * ns, config.sensor_dims_per_activist)
end

# Additive-bias sensor: base noise plus a constant offset of magnitude
# config.drift_sensor_scale on every observed dimension. Used as a GROUND-TRUTH
# sensor model to inject a persistent, unfilterable observation bias that the
# observing player's own filter (built from a bias-free model) never corrects.
function drift_sensor_model(x::BlockVector, ns::BlockVector; config::PlayerConfig)
    BlockVector(
        x + config.sensor_noise_scale * ns +
            config.drift_sensor_scale * ones(length(x)),
        config.sensor_dims_per_activist)
end

function covariance_drift_sensor_model(x::BlockVector, ns::BlockVector; config::PlayerConfig)
    BlockVector(
        x + (config.sensor_noise_scale + config.drift_sensor_scale) * ns,
        config.sensor_dims_per_activist
    )
end