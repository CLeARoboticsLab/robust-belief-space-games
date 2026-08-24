# IMPORTANT: this file must stay the LAST include in Senate.jl, and new
# functions must only be APPENDED here. Serialized run archives (.dat) contain
# closures whose gensym'd names (#N#M) depend on the module's lowering counter;
# defining a new function (kwargs/closures consume counter slots) anywhere
# before existing definitions renumbers them and makes every old archive fail
# to deserialize with UndefVarError(Symbol("#N#M"), Senate).

# Directional drift along the goal-separation axis (P2's target [1,3] minus P1's
# target [3,1], normalized). Every senator's state picks up a constant per-step
# offset of signed magnitude config.drift_dynamics_scale in that direction.
# Positive scale drifts believed senators toward the up-left, so the drifting
# player's effective push target slides TOWARD the opponent's goal; negative
# slides it away. Direction is a constant (not a config field) so PlayerConfig's
# position-based custom deserializer keeps reading old archives.
const GOALLINE_DRIFT_DIRECTION = [-1.0, 1.0] / sqrt(2.0)
function goalline_drift_dynamics_model(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig, undrifted_dynamics_model::Function=base_dynamics)
    BlockVector(
        undrifted_dynamics_model(x, u, m; config=config) +
        config.drift_dynamics_scale * repeat(GOALLINE_DRIFT_DIRECTION, length(x) ÷ 2),
        length.(x.blocks))
end
