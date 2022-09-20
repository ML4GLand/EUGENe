:github_url: eugene.models.Jores21CNN

eugene.models.Jores21CNN
========================

.. currentmodule:: eugene.models

.. autoclass:: Jores21CNN



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.Jores21CNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.Jores21CNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.Jores21CNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.Jores21CNN.T_destination
      ~eugene.models.Jores21CNN.automatic_optimization
      ~eugene.models.Jores21CNN.current_epoch
      ~eugene.models.Jores21CNN.device
      ~eugene.models.Jores21CNN.dtype
      ~eugene.models.Jores21CNN.dump_patches
      ~eugene.models.Jores21CNN.example_input_array
      ~eugene.models.Jores21CNN.global_rank
      ~eugene.models.Jores21CNN.global_step
      ~eugene.models.Jores21CNN.hparams
      ~eugene.models.Jores21CNN.hparams_initial
      ~eugene.models.Jores21CNN.loaded_optimizer_states_dict
      ~eugene.models.Jores21CNN.local_rank
      ~eugene.models.Jores21CNN.logger
      ~eugene.models.Jores21CNN.model_size
      ~eugene.models.Jores21CNN.on_gpu
      ~eugene.models.Jores21CNN.truncated_bptt_steps
      ~eugene.models.Jores21CNN.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.Jores21CNN.add_module
      ~eugene.models.Jores21CNN.add_to_queue
      ~eugene.models.Jores21CNN.all_gather
      ~eugene.models.Jores21CNN.apply
      ~eugene.models.Jores21CNN.backward
      ~eugene.models.Jores21CNN.bfloat16
      ~eugene.models.Jores21CNN.buffers
      ~eugene.models.Jores21CNN.children
      ~eugene.models.Jores21CNN.clip_gradients
      ~eugene.models.Jores21CNN.configure_callbacks
      ~eugene.models.Jores21CNN.configure_gradient_clipping
      ~eugene.models.Jores21CNN.configure_optimizers
      ~eugene.models.Jores21CNN.configure_sharded_model
      ~eugene.models.Jores21CNN.cpu
      ~eugene.models.Jores21CNN.cuda
      ~eugene.models.Jores21CNN.double
      ~eugene.models.Jores21CNN.eval
      ~eugene.models.Jores21CNN.extra_repr
      ~eugene.models.Jores21CNN.float
      ~eugene.models.Jores21CNN.forward
      ~eugene.models.Jores21CNN.freeze
      ~eugene.models.Jores21CNN.get_buffer
      ~eugene.models.Jores21CNN.get_extra_state
      ~eugene.models.Jores21CNN.get_from_queue
      ~eugene.models.Jores21CNN.get_parameter
      ~eugene.models.Jores21CNN.get_progress_bar_dict
      ~eugene.models.Jores21CNN.get_submodule
      ~eugene.models.Jores21CNN.half
      ~eugene.models.Jores21CNN.load_from_checkpoint
      ~eugene.models.Jores21CNN.load_state_dict
      ~eugene.models.Jores21CNN.log
      ~eugene.models.Jores21CNN.log_dict
      ~eugene.models.Jores21CNN.log_grad_norm
      ~eugene.models.Jores21CNN.lr_schedulers
      ~eugene.models.Jores21CNN.manual_backward
      ~eugene.models.Jores21CNN.modules
      ~eugene.models.Jores21CNN.named_buffers
      ~eugene.models.Jores21CNN.named_children
      ~eugene.models.Jores21CNN.named_modules
      ~eugene.models.Jores21CNN.named_parameters
      ~eugene.models.Jores21CNN.on_after_backward
      ~eugene.models.Jores21CNN.on_after_batch_transfer
      ~eugene.models.Jores21CNN.on_before_backward
      ~eugene.models.Jores21CNN.on_before_batch_transfer
      ~eugene.models.Jores21CNN.on_before_optimizer_step
      ~eugene.models.Jores21CNN.on_before_zero_grad
      ~eugene.models.Jores21CNN.on_epoch_end
      ~eugene.models.Jores21CNN.on_epoch_start
      ~eugene.models.Jores21CNN.on_fit_end
      ~eugene.models.Jores21CNN.on_fit_start
      ~eugene.models.Jores21CNN.on_hpc_load
      ~eugene.models.Jores21CNN.on_hpc_save
      ~eugene.models.Jores21CNN.on_load_checkpoint
      ~eugene.models.Jores21CNN.on_post_move_to_device
      ~eugene.models.Jores21CNN.on_predict_batch_end
      ~eugene.models.Jores21CNN.on_predict_batch_start
      ~eugene.models.Jores21CNN.on_predict_dataloader
      ~eugene.models.Jores21CNN.on_predict_end
      ~eugene.models.Jores21CNN.on_predict_epoch_end
      ~eugene.models.Jores21CNN.on_predict_epoch_start
      ~eugene.models.Jores21CNN.on_predict_model_eval
      ~eugene.models.Jores21CNN.on_predict_start
      ~eugene.models.Jores21CNN.on_pretrain_routine_end
      ~eugene.models.Jores21CNN.on_pretrain_routine_start
      ~eugene.models.Jores21CNN.on_save_checkpoint
      ~eugene.models.Jores21CNN.on_test_batch_end
      ~eugene.models.Jores21CNN.on_test_batch_start
      ~eugene.models.Jores21CNN.on_test_dataloader
      ~eugene.models.Jores21CNN.on_test_end
      ~eugene.models.Jores21CNN.on_test_epoch_end
      ~eugene.models.Jores21CNN.on_test_epoch_start
      ~eugene.models.Jores21CNN.on_test_model_eval
      ~eugene.models.Jores21CNN.on_test_model_train
      ~eugene.models.Jores21CNN.on_test_start
      ~eugene.models.Jores21CNN.on_train_batch_end
      ~eugene.models.Jores21CNN.on_train_batch_start
      ~eugene.models.Jores21CNN.on_train_dataloader
      ~eugene.models.Jores21CNN.on_train_end
      ~eugene.models.Jores21CNN.on_train_epoch_end
      ~eugene.models.Jores21CNN.on_train_epoch_start
      ~eugene.models.Jores21CNN.on_train_start
      ~eugene.models.Jores21CNN.on_val_dataloader
      ~eugene.models.Jores21CNN.on_validation_batch_end
      ~eugene.models.Jores21CNN.on_validation_batch_start
      ~eugene.models.Jores21CNN.on_validation_end
      ~eugene.models.Jores21CNN.on_validation_epoch_end
      ~eugene.models.Jores21CNN.on_validation_epoch_start
      ~eugene.models.Jores21CNN.on_validation_model_eval
      ~eugene.models.Jores21CNN.on_validation_model_train
      ~eugene.models.Jores21CNN.on_validation_start
      ~eugene.models.Jores21CNN.optimizer_step
      ~eugene.models.Jores21CNN.optimizer_zero_grad
      ~eugene.models.Jores21CNN.optimizers
      ~eugene.models.Jores21CNN.parameters
      ~eugene.models.Jores21CNN.predict_dataloader
      ~eugene.models.Jores21CNN.predict_step
      ~eugene.models.Jores21CNN.prepare_data
      ~eugene.models.Jores21CNN.print
      ~eugene.models.Jores21CNN.register_backward_hook
      ~eugene.models.Jores21CNN.register_buffer
      ~eugene.models.Jores21CNN.register_forward_hook
      ~eugene.models.Jores21CNN.register_forward_pre_hook
      ~eugene.models.Jores21CNN.register_full_backward_hook
      ~eugene.models.Jores21CNN.register_module
      ~eugene.models.Jores21CNN.register_parameter
      ~eugene.models.Jores21CNN.requires_grad_
      ~eugene.models.Jores21CNN.save_hyperparameters
      ~eugene.models.Jores21CNN.set_extra_state
      ~eugene.models.Jores21CNN.setup
      ~eugene.models.Jores21CNN.share_memory
      ~eugene.models.Jores21CNN.state_dict
      ~eugene.models.Jores21CNN.summarize
      ~eugene.models.Jores21CNN.summary
      ~eugene.models.Jores21CNN.tbptt_split_batch
      ~eugene.models.Jores21CNN.teardown
      ~eugene.models.Jores21CNN.test_dataloader
      ~eugene.models.Jores21CNN.test_epoch_end
      ~eugene.models.Jores21CNN.test_step
      ~eugene.models.Jores21CNN.test_step_end
      ~eugene.models.Jores21CNN.to
      ~eugene.models.Jores21CNN.to_empty
      ~eugene.models.Jores21CNN.to_onnx
      ~eugene.models.Jores21CNN.to_torchscript
      ~eugene.models.Jores21CNN.toggle_optimizer
      ~eugene.models.Jores21CNN.train
      ~eugene.models.Jores21CNN.train_dataloader
      ~eugene.models.Jores21CNN.training_epoch_end
      ~eugene.models.Jores21CNN.training_step
      ~eugene.models.Jores21CNN.training_step_end
      ~eugene.models.Jores21CNN.transfer_batch_to_device
      ~eugene.models.Jores21CNN.type
      ~eugene.models.Jores21CNN.unfreeze
      ~eugene.models.Jores21CNN.untoggle_optimizer
      ~eugene.models.Jores21CNN.val_dataloader
      ~eugene.models.Jores21CNN.validation_epoch_end
      ~eugene.models.Jores21CNN.validation_step
      ~eugene.models.Jores21CNN.validation_step_end
      ~eugene.models.Jores21CNN.xpu
      ~eugene.models.Jores21CNN.zero_grad



.. _sphx_glr_backref_eugene.models.Jores21CNN:
