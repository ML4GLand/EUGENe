:github_url: eugene.models.Hybrid

eugene.models.Hybrid
====================

.. currentmodule:: eugene.models

.. autoclass:: Hybrid



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.Hybrid.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.Hybrid.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.Hybrid.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.Hybrid.T_destination
      ~eugene.models.Hybrid.automatic_optimization
      ~eugene.models.Hybrid.current_epoch
      ~eugene.models.Hybrid.device
      ~eugene.models.Hybrid.dtype
      ~eugene.models.Hybrid.dump_patches
      ~eugene.models.Hybrid.example_input_array
      ~eugene.models.Hybrid.global_rank
      ~eugene.models.Hybrid.global_step
      ~eugene.models.Hybrid.hparams
      ~eugene.models.Hybrid.hparams_initial
      ~eugene.models.Hybrid.loaded_optimizer_states_dict
      ~eugene.models.Hybrid.local_rank
      ~eugene.models.Hybrid.logger
      ~eugene.models.Hybrid.model_size
      ~eugene.models.Hybrid.on_gpu
      ~eugene.models.Hybrid.truncated_bptt_steps
      ~eugene.models.Hybrid.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.Hybrid.add_module
      ~eugene.models.Hybrid.add_to_queue
      ~eugene.models.Hybrid.all_gather
      ~eugene.models.Hybrid.apply
      ~eugene.models.Hybrid.backward
      ~eugene.models.Hybrid.bfloat16
      ~eugene.models.Hybrid.buffers
      ~eugene.models.Hybrid.children
      ~eugene.models.Hybrid.clip_gradients
      ~eugene.models.Hybrid.configure_callbacks
      ~eugene.models.Hybrid.configure_gradient_clipping
      ~eugene.models.Hybrid.configure_optimizers
      ~eugene.models.Hybrid.configure_sharded_model
      ~eugene.models.Hybrid.cpu
      ~eugene.models.Hybrid.cuda
      ~eugene.models.Hybrid.double
      ~eugene.models.Hybrid.eval
      ~eugene.models.Hybrid.extra_repr
      ~eugene.models.Hybrid.float
      ~eugene.models.Hybrid.forward
      ~eugene.models.Hybrid.freeze
      ~eugene.models.Hybrid.get_buffer
      ~eugene.models.Hybrid.get_extra_state
      ~eugene.models.Hybrid.get_from_queue
      ~eugene.models.Hybrid.get_parameter
      ~eugene.models.Hybrid.get_progress_bar_dict
      ~eugene.models.Hybrid.get_submodule
      ~eugene.models.Hybrid.half
      ~eugene.models.Hybrid.load_from_checkpoint
      ~eugene.models.Hybrid.load_state_dict
      ~eugene.models.Hybrid.log
      ~eugene.models.Hybrid.log_dict
      ~eugene.models.Hybrid.log_grad_norm
      ~eugene.models.Hybrid.lr_schedulers
      ~eugene.models.Hybrid.manual_backward
      ~eugene.models.Hybrid.modules
      ~eugene.models.Hybrid.named_buffers
      ~eugene.models.Hybrid.named_children
      ~eugene.models.Hybrid.named_modules
      ~eugene.models.Hybrid.named_parameters
      ~eugene.models.Hybrid.on_after_backward
      ~eugene.models.Hybrid.on_after_batch_transfer
      ~eugene.models.Hybrid.on_before_backward
      ~eugene.models.Hybrid.on_before_batch_transfer
      ~eugene.models.Hybrid.on_before_optimizer_step
      ~eugene.models.Hybrid.on_before_zero_grad
      ~eugene.models.Hybrid.on_epoch_end
      ~eugene.models.Hybrid.on_epoch_start
      ~eugene.models.Hybrid.on_fit_end
      ~eugene.models.Hybrid.on_fit_start
      ~eugene.models.Hybrid.on_hpc_load
      ~eugene.models.Hybrid.on_hpc_save
      ~eugene.models.Hybrid.on_load_checkpoint
      ~eugene.models.Hybrid.on_post_move_to_device
      ~eugene.models.Hybrid.on_predict_batch_end
      ~eugene.models.Hybrid.on_predict_batch_start
      ~eugene.models.Hybrid.on_predict_dataloader
      ~eugene.models.Hybrid.on_predict_end
      ~eugene.models.Hybrid.on_predict_epoch_end
      ~eugene.models.Hybrid.on_predict_epoch_start
      ~eugene.models.Hybrid.on_predict_model_eval
      ~eugene.models.Hybrid.on_predict_start
      ~eugene.models.Hybrid.on_pretrain_routine_end
      ~eugene.models.Hybrid.on_pretrain_routine_start
      ~eugene.models.Hybrid.on_save_checkpoint
      ~eugene.models.Hybrid.on_test_batch_end
      ~eugene.models.Hybrid.on_test_batch_start
      ~eugene.models.Hybrid.on_test_dataloader
      ~eugene.models.Hybrid.on_test_end
      ~eugene.models.Hybrid.on_test_epoch_end
      ~eugene.models.Hybrid.on_test_epoch_start
      ~eugene.models.Hybrid.on_test_model_eval
      ~eugene.models.Hybrid.on_test_model_train
      ~eugene.models.Hybrid.on_test_start
      ~eugene.models.Hybrid.on_train_batch_end
      ~eugene.models.Hybrid.on_train_batch_start
      ~eugene.models.Hybrid.on_train_dataloader
      ~eugene.models.Hybrid.on_train_end
      ~eugene.models.Hybrid.on_train_epoch_end
      ~eugene.models.Hybrid.on_train_epoch_start
      ~eugene.models.Hybrid.on_train_start
      ~eugene.models.Hybrid.on_val_dataloader
      ~eugene.models.Hybrid.on_validation_batch_end
      ~eugene.models.Hybrid.on_validation_batch_start
      ~eugene.models.Hybrid.on_validation_end
      ~eugene.models.Hybrid.on_validation_epoch_end
      ~eugene.models.Hybrid.on_validation_epoch_start
      ~eugene.models.Hybrid.on_validation_model_eval
      ~eugene.models.Hybrid.on_validation_model_train
      ~eugene.models.Hybrid.on_validation_start
      ~eugene.models.Hybrid.optimizer_step
      ~eugene.models.Hybrid.optimizer_zero_grad
      ~eugene.models.Hybrid.optimizers
      ~eugene.models.Hybrid.parameters
      ~eugene.models.Hybrid.predict_dataloader
      ~eugene.models.Hybrid.predict_step
      ~eugene.models.Hybrid.prepare_data
      ~eugene.models.Hybrid.print
      ~eugene.models.Hybrid.register_backward_hook
      ~eugene.models.Hybrid.register_buffer
      ~eugene.models.Hybrid.register_forward_hook
      ~eugene.models.Hybrid.register_forward_pre_hook
      ~eugene.models.Hybrid.register_full_backward_hook
      ~eugene.models.Hybrid.register_module
      ~eugene.models.Hybrid.register_parameter
      ~eugene.models.Hybrid.requires_grad_
      ~eugene.models.Hybrid.save_hyperparameters
      ~eugene.models.Hybrid.set_extra_state
      ~eugene.models.Hybrid.setup
      ~eugene.models.Hybrid.share_memory
      ~eugene.models.Hybrid.state_dict
      ~eugene.models.Hybrid.summarize
      ~eugene.models.Hybrid.summary
      ~eugene.models.Hybrid.tbptt_split_batch
      ~eugene.models.Hybrid.teardown
      ~eugene.models.Hybrid.test_dataloader
      ~eugene.models.Hybrid.test_epoch_end
      ~eugene.models.Hybrid.test_step
      ~eugene.models.Hybrid.test_step_end
      ~eugene.models.Hybrid.to
      ~eugene.models.Hybrid.to_empty
      ~eugene.models.Hybrid.to_onnx
      ~eugene.models.Hybrid.to_torchscript
      ~eugene.models.Hybrid.toggle_optimizer
      ~eugene.models.Hybrid.train
      ~eugene.models.Hybrid.train_dataloader
      ~eugene.models.Hybrid.training_epoch_end
      ~eugene.models.Hybrid.training_step
      ~eugene.models.Hybrid.training_step_end
      ~eugene.models.Hybrid.transfer_batch_to_device
      ~eugene.models.Hybrid.type
      ~eugene.models.Hybrid.unfreeze
      ~eugene.models.Hybrid.untoggle_optimizer
      ~eugene.models.Hybrid.val_dataloader
      ~eugene.models.Hybrid.validation_epoch_end
      ~eugene.models.Hybrid.validation_step
      ~eugene.models.Hybrid.validation_step_end
      ~eugene.models.Hybrid.xpu
      ~eugene.models.Hybrid.zero_grad



.. _sphx_glr_backref_eugene.models.Hybrid:
