target_include_directories(muzero PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_sources(muzero
    PRIVATE 
        default_flags.cpp
        config.cpp
        game_history.cpp
        learner.cpp
        mcts.cpp
        metric_logger.cpp
        model_layers.cpp
        models.cpp
        muzero.cpp
        replay_buffer.cpp
        self_play.cpp
        signaller.cpp
        util.cpp
        value_encoder.cpp
        vprnet.cpp
        vprnet_evaluator.cpp
)
