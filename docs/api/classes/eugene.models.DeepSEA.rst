:github_url: eugene.models.DeepSEA

eugene.models.DeepSEA
=====================

.. currentmodule:: eugene.models

.. autoclass:: DeepSEA



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.DeepSEA.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.DeepSEA.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.DeepSEA.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.DeepSEA.T_destination
      ~eugene.models.DeepSEA.automatic_optimization
      ~eugene.models.DeepSEA.current_epoch
      ~eugene.models.DeepSEA.device
      ~eugene.models.DeepSEA.dtype
      ~eugene.models.DeepSEA.dump_patches
      ~eugene.models.DeepSEA.example_input_array
      ~eugene.models.DeepSEA.global_rank
      ~eugene.models.DeepSEA.global_step
      ~eugene.models.DeepSEA.hparams
      ~eugene.models.DeepSEA.hparams_initial
      ~eugene.models.DeepSEA.loaded_optimizer_states_dict
      ~eugene.models.DeepSEA.local_rank
      ~eugene.models.DeepSEA.logger
      ~eugene.models.DeepSEA.model_size
      ~eugene.models.DeepSEA.on_gpu
      ~eugene.models.DeepSEA.truncated_bptt_steps
      ~eugene.models.DeepSEA.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.DeepSEA.add_module
      ~eugene.models.DeepSEA.add_to_queue
      ~eugene.models.DeepSEA.all_gather
      ~eugene.models.DeepSEA.apply
      ~eugene.models.DeepSEA.backward
      ~eugene.models.DeepSEA.bfloat16
      ~eugene.models.DeepSEA.buffers
      ~eugene.models.DeepSEA.children
      ~eugene.models.DeepSEA.clip_gradients
      ~eugene.models.DeepSEA.configure_callbacks
      ~eugene.models.DeepSEA.configure_gradient_clipping
      ~eugene.models.DeepSEA.configure_optimizers
      ~eugene.models.DeepSEA.configure_sharded_model
      ~eugene.models.DeepSEA.cpu
      ~eugene.models.DeepSEA.cuda
      ~eugene.models.DeepSEA.double
      ~eugene.models.DeepSEA.eval
      ~eugene.models.DeepSEA.extra_repr
      ~eugene.models.DeepSEA.float
      ~eugene.models.DeepSEA.forward
      ~eugene.models.DeepSEA.freeze
      ~eugene.models.DeepSEA.get_buffer
      ~eugene.models.DeepSEA.get_extra_state
      ~eugene.models.DeepSEA.get_from_queue
      ~eugene.models.DeepSEA.get_parameter
      ~eugene.models.DeepSEA.get_progress_bar_dict
      ~eugene.models.DeepSEA.get_submodule
      ~eugene.models.DeepSEA.half
      ~eugene.models.DeepSEA.kwarg_handler
      ~eugene.models.DeepSEA.load_from_checkpoint
      ~eugene.models.DeepSEA.load_state_dict
      ~eugene.models.DeepSEA.log
      ~eugene.models.DeepSEA.log_dict
      ~eugene.models.DeepSEA.log_grad_norm
      ~eugene.models.DeepSEA.lr_schedulers
      ~eugene.models.DeepSEA.manual_backward
      ~eugene.models.DeepSEA.modules
      ~eugene.models.DeepSEA.named_buffers
      ~eugene.models.DeepSEA.named_children
      ~eugene.models.DeepSEA.named_modules
      ~eugene.models.DeepSEA.named_parameters
      ~eugene.models.DeepSEA.on_after_backward
      ~eugene.models.DeepSEA.on_after_batch_transfer
      ~eugene.models.DeepSEA.on_before_backward
      ~eugene.models.DeepSEA.on_before_batch_transfer
      ~eugene.models.DeepSEA.on_before_optimizer_step
      ~eugene.models.DeepSEA.on_before_zero_grad
      ~eugene.models.DeepSEA.on_epoch_end
      ~eugene.models.DeepSEA.on_epoch_start
      ~eugene.models.DeepSEA.on_fit_end
      ~eugene.models.DeepSEA.on_fit_start
      ~eugene.models.DeepSEA.on_hpc_load
      ~eugene.models.DeepSEA.on_hpc_save
      ~eugene.models.DeepSEA.on_load_checkpoint
      ~eugene.models.DeepSEA.on_post_move_to_device
      ~eugene.models.DeepSEA.on_predict_batch_end
      ~eugene.models.DeepSEA.on_predict_batch_start
      ~eugene.models.DeepSEA.on_predict_dataloader
      ~eugene.models.DeepSEA.on_predict_end
      ~eugene.models.DeepSEA.on_predict_epoch_end
      ~eugene.models.DeepSEA.on_predict_epoch_start
      ~eugene.models.DeepSEA.on_predict_model_eval
      ~eugene.models.DeepSEA.on_predict_start
      ~eugene.models.DeepSEA.on_pretrain_routine_end
      ~eugene.models.DeepSEA.on_pretrain_routine_start
      ~eugene.models.DeepSEA.on_save_checkpoint
      ~eugene.models.DeepSEA.on_test_batch_end
      ~eugene.models.DeepSEA.on_test_batch_start
      ~eugene.models.DeepSEA.on_test_dataloader
      ~eugene.models.DeepSEA.on_test_end
      ~eugene.models.DeepSEA.on_test_epoch_end
      ~eugene.models.DeepSEA.on_test_epoch_start
      ~eugene.models.DeepSEA.on_test_model_eval
      ~eugene.models.DeepSEA.on_test_model_train
      ~eugene.models.DeepSEA.on_test_start
      ~eugene.models.DeepSEA.on_train_batch_end
      ~eugene.models.DeepSEA.on_train_batch_start
      ~eugene.models.DeepSEA.on_train_dataloader
      ~eugene.models.DeepSEA.on_train_end
      ~eugene.models.DeepSEA.on_train_epoch_end
      ~eugene.models.DeepSEA.on_train_epoch_start
      ~eugene.models.DeepSEA.on_train_start
      ~eugene.models.DeepSEA.on_val_dataloader
      ~eugene.models.DeepSEA.on_validation_batch_end
      ~eugene.models.DeepSEA.on_validation_batch_start
      ~eugene.models.DeepSEA.on_validation_end
      ~eugene.models.DeepSEA.on_validation_epoch_end
      ~eugene.models.DeepSEA.on_validation_epoch_start
      ~eugene.models.DeepSEA.on_validation_model_eval
      ~eugene.models.DeepSEA.on_validation_model_train
      ~eugene.models.DeepSEA.on_validation_start
      ~eugene.models.DeepSEA.optimizer_step
      ~eugene.models.DeepSEA.optimizer_zero_grad
      ~eugene.models.DeepSEA.optimizers
      ~eugene.models.DeepSEA.parameters
      ~eugene.models.DeepSEA.predict_dataloader
      ~eugene.models.DeepSEA.predict_step
      ~eugene.models.DeepSEA.prepare_data
      ~eugene.models.DeepSEA.print
      ~eugene.models.DeepSEA.register_backward_hook
      ~eugene.models.DeepSEA.register_buffer
      ~eugene.models.DeepSEA.register_forward_hook
      ~eugene.models.DeepSEA.register_forward_pre_hook
      ~eugene.models.DeepSEA.register_full_backward_hook
      ~eugene.models.DeepSEA.register_module
      ~eugene.models.DeepSEA.register_parameter
      ~eugene.models.DeepSEA.requires_grad_
      ~eugene.models.DeepSEA.save_hyperparameters
      ~eugene.models.DeepSEA.set_extra_state
      ~eugene.models.DeepSEA.setup
      ~eugene.models.DeepSEA.share_memory
      ~eugene.models.DeepSEA.state_dict
      ~eugene.models.DeepSEA.summarize
      ~eugene.models.DeepSEA.summary
      ~eugene.models.DeepSEA.tbptt_split_batch
      ~eugene.models.DeepSEA.teardown
      ~eugene.models.DeepSEA.test_dataloader
      ~eugene.models.DeepSEA.test_epoch_end
      ~eugene.models.DeepSEA.test_step
      ~eugene.models.DeepSEA.test_step_end
      ~eugene.models.DeepSEA.to
      ~eugene.models.DeepSEA.to_empty
      ~eugene.models.DeepSEA.to_onnx
      ~eugene.models.DeepSEA.to_torchscript
      ~eugene.models.DeepSEA.toggle_optimizer
      ~eugene.models.DeepSEA.train
      ~eugene.models.DeepSEA.train_dataloader
      ~eugene.models.DeepSEA.training_epoch_end
      ~eugene.models.DeepSEA.training_step
      ~eugene.models.DeepSEA.training_step_end
      ~eugene.models.DeepSEA.transfer_batch_to_device
      ~eugene.models.DeepSEA.type
      ~eugene.models.DeepSEA.unfreeze
      ~eugene.models.DeepSEA.untoggle_optimizer
      ~eugene.models.DeepSEA.val_dataloader
      ~eugene.models.DeepSEA.validation_epoch_end
      ~eugene.models.DeepSEA.validation_step
      ~eugene.models.DeepSEA.validation_step_end
      ~eugene.models.DeepSEA.xpu
      ~eugene.models.DeepSEA.zero_grad



.. _sphx_glr_backref_eugene.models.DeepSEA:
