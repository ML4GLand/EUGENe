:github_url: eugene.models.DeepBind

eugene.models.DeepBind
======================

.. currentmodule:: eugene.models

.. autoclass:: DeepBind



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.DeepBind.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.DeepBind.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.DeepBind.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.DeepBind.T_destination
      ~eugene.models.DeepBind.automatic_optimization
      ~eugene.models.DeepBind.current_epoch
      ~eugene.models.DeepBind.device
      ~eugene.models.DeepBind.dtype
      ~eugene.models.DeepBind.dump_patches
      ~eugene.models.DeepBind.example_input_array
      ~eugene.models.DeepBind.global_rank
      ~eugene.models.DeepBind.global_step
      ~eugene.models.DeepBind.hparams
      ~eugene.models.DeepBind.hparams_initial
      ~eugene.models.DeepBind.loaded_optimizer_states_dict
      ~eugene.models.DeepBind.local_rank
      ~eugene.models.DeepBind.logger
      ~eugene.models.DeepBind.model_size
      ~eugene.models.DeepBind.on_gpu
      ~eugene.models.DeepBind.truncated_bptt_steps
      ~eugene.models.DeepBind.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.DeepBind.add_module
      ~eugene.models.DeepBind.add_to_queue
      ~eugene.models.DeepBind.all_gather
      ~eugene.models.DeepBind.apply
      ~eugene.models.DeepBind.backward
      ~eugene.models.DeepBind.bfloat16
      ~eugene.models.DeepBind.buffers
      ~eugene.models.DeepBind.children
      ~eugene.models.DeepBind.clip_gradients
      ~eugene.models.DeepBind.configure_callbacks
      ~eugene.models.DeepBind.configure_gradient_clipping
      ~eugene.models.DeepBind.configure_optimizers
      ~eugene.models.DeepBind.configure_sharded_model
      ~eugene.models.DeepBind.cpu
      ~eugene.models.DeepBind.cuda
      ~eugene.models.DeepBind.double
      ~eugene.models.DeepBind.eval
      ~eugene.models.DeepBind.extra_repr
      ~eugene.models.DeepBind.float
      ~eugene.models.DeepBind.forward
      ~eugene.models.DeepBind.freeze
      ~eugene.models.DeepBind.get_buffer
      ~eugene.models.DeepBind.get_extra_state
      ~eugene.models.DeepBind.get_from_queue
      ~eugene.models.DeepBind.get_parameter
      ~eugene.models.DeepBind.get_progress_bar_dict
      ~eugene.models.DeepBind.get_submodule
      ~eugene.models.DeepBind.half
      ~eugene.models.DeepBind.kwarg_handler
      ~eugene.models.DeepBind.load_from_checkpoint
      ~eugene.models.DeepBind.load_state_dict
      ~eugene.models.DeepBind.log
      ~eugene.models.DeepBind.log_dict
      ~eugene.models.DeepBind.log_grad_norm
      ~eugene.models.DeepBind.lr_schedulers
      ~eugene.models.DeepBind.manual_backward
      ~eugene.models.DeepBind.modules
      ~eugene.models.DeepBind.named_buffers
      ~eugene.models.DeepBind.named_children
      ~eugene.models.DeepBind.named_modules
      ~eugene.models.DeepBind.named_parameters
      ~eugene.models.DeepBind.on_after_backward
      ~eugene.models.DeepBind.on_after_batch_transfer
      ~eugene.models.DeepBind.on_before_backward
      ~eugene.models.DeepBind.on_before_batch_transfer
      ~eugene.models.DeepBind.on_before_optimizer_step
      ~eugene.models.DeepBind.on_before_zero_grad
      ~eugene.models.DeepBind.on_epoch_end
      ~eugene.models.DeepBind.on_epoch_start
      ~eugene.models.DeepBind.on_fit_end
      ~eugene.models.DeepBind.on_fit_start
      ~eugene.models.DeepBind.on_hpc_load
      ~eugene.models.DeepBind.on_hpc_save
      ~eugene.models.DeepBind.on_load_checkpoint
      ~eugene.models.DeepBind.on_post_move_to_device
      ~eugene.models.DeepBind.on_predict_batch_end
      ~eugene.models.DeepBind.on_predict_batch_start
      ~eugene.models.DeepBind.on_predict_dataloader
      ~eugene.models.DeepBind.on_predict_end
      ~eugene.models.DeepBind.on_predict_epoch_end
      ~eugene.models.DeepBind.on_predict_epoch_start
      ~eugene.models.DeepBind.on_predict_model_eval
      ~eugene.models.DeepBind.on_predict_start
      ~eugene.models.DeepBind.on_pretrain_routine_end
      ~eugene.models.DeepBind.on_pretrain_routine_start
      ~eugene.models.DeepBind.on_save_checkpoint
      ~eugene.models.DeepBind.on_test_batch_end
      ~eugene.models.DeepBind.on_test_batch_start
      ~eugene.models.DeepBind.on_test_dataloader
      ~eugene.models.DeepBind.on_test_end
      ~eugene.models.DeepBind.on_test_epoch_end
      ~eugene.models.DeepBind.on_test_epoch_start
      ~eugene.models.DeepBind.on_test_model_eval
      ~eugene.models.DeepBind.on_test_model_train
      ~eugene.models.DeepBind.on_test_start
      ~eugene.models.DeepBind.on_train_batch_end
      ~eugene.models.DeepBind.on_train_batch_start
      ~eugene.models.DeepBind.on_train_dataloader
      ~eugene.models.DeepBind.on_train_end
      ~eugene.models.DeepBind.on_train_epoch_end
      ~eugene.models.DeepBind.on_train_epoch_start
      ~eugene.models.DeepBind.on_train_start
      ~eugene.models.DeepBind.on_val_dataloader
      ~eugene.models.DeepBind.on_validation_batch_end
      ~eugene.models.DeepBind.on_validation_batch_start
      ~eugene.models.DeepBind.on_validation_end
      ~eugene.models.DeepBind.on_validation_epoch_end
      ~eugene.models.DeepBind.on_validation_epoch_start
      ~eugene.models.DeepBind.on_validation_model_eval
      ~eugene.models.DeepBind.on_validation_model_train
      ~eugene.models.DeepBind.on_validation_start
      ~eugene.models.DeepBind.optimizer_step
      ~eugene.models.DeepBind.optimizer_zero_grad
      ~eugene.models.DeepBind.optimizers
      ~eugene.models.DeepBind.parameters
      ~eugene.models.DeepBind.predict_dataloader
      ~eugene.models.DeepBind.predict_step
      ~eugene.models.DeepBind.prepare_data
      ~eugene.models.DeepBind.print
      ~eugene.models.DeepBind.register_backward_hook
      ~eugene.models.DeepBind.register_buffer
      ~eugene.models.DeepBind.register_forward_hook
      ~eugene.models.DeepBind.register_forward_pre_hook
      ~eugene.models.DeepBind.register_full_backward_hook
      ~eugene.models.DeepBind.register_module
      ~eugene.models.DeepBind.register_parameter
      ~eugene.models.DeepBind.requires_grad_
      ~eugene.models.DeepBind.save_hyperparameters
      ~eugene.models.DeepBind.set_extra_state
      ~eugene.models.DeepBind.setup
      ~eugene.models.DeepBind.share_memory
      ~eugene.models.DeepBind.state_dict
      ~eugene.models.DeepBind.summarize
      ~eugene.models.DeepBind.summary
      ~eugene.models.DeepBind.tbptt_split_batch
      ~eugene.models.DeepBind.teardown
      ~eugene.models.DeepBind.test_dataloader
      ~eugene.models.DeepBind.test_epoch_end
      ~eugene.models.DeepBind.test_step
      ~eugene.models.DeepBind.test_step_end
      ~eugene.models.DeepBind.to
      ~eugene.models.DeepBind.to_empty
      ~eugene.models.DeepBind.to_onnx
      ~eugene.models.DeepBind.to_torchscript
      ~eugene.models.DeepBind.toggle_optimizer
      ~eugene.models.DeepBind.train
      ~eugene.models.DeepBind.train_dataloader
      ~eugene.models.DeepBind.training_epoch_end
      ~eugene.models.DeepBind.training_step
      ~eugene.models.DeepBind.training_step_end
      ~eugene.models.DeepBind.transfer_batch_to_device
      ~eugene.models.DeepBind.type
      ~eugene.models.DeepBind.unfreeze
      ~eugene.models.DeepBind.untoggle_optimizer
      ~eugene.models.DeepBind.val_dataloader
      ~eugene.models.DeepBind.validation_epoch_end
      ~eugene.models.DeepBind.validation_step
      ~eugene.models.DeepBind.validation_step_end
      ~eugene.models.DeepBind.xpu
      ~eugene.models.DeepBind.zero_grad



.. _sphx_glr_backref_eugene.models.DeepBind:
