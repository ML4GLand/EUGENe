:github_url: eugene.models.CNN

eugene.models.CNN
=================

.. currentmodule:: eugene.models

.. autoclass:: CNN



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.CNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.CNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.CNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.CNN.T_destination
      ~eugene.models.CNN.automatic_optimization
      ~eugene.models.CNN.current_epoch
      ~eugene.models.CNN.device
      ~eugene.models.CNN.dtype
      ~eugene.models.CNN.dump_patches
      ~eugene.models.CNN.example_input_array
      ~eugene.models.CNN.global_rank
      ~eugene.models.CNN.global_step
      ~eugene.models.CNN.hparams
      ~eugene.models.CNN.hparams_initial
      ~eugene.models.CNN.loaded_optimizer_states_dict
      ~eugene.models.CNN.local_rank
      ~eugene.models.CNN.logger
      ~eugene.models.CNN.model_size
      ~eugene.models.CNN.on_gpu
      ~eugene.models.CNN.truncated_bptt_steps
      ~eugene.models.CNN.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.CNN.add_module
      ~eugene.models.CNN.add_to_queue
      ~eugene.models.CNN.all_gather
      ~eugene.models.CNN.apply
      ~eugene.models.CNN.backward
      ~eugene.models.CNN.bfloat16
      ~eugene.models.CNN.buffers
      ~eugene.models.CNN.children
      ~eugene.models.CNN.clip_gradients
      ~eugene.models.CNN.configure_callbacks
      ~eugene.models.CNN.configure_gradient_clipping
      ~eugene.models.CNN.configure_optimizers
      ~eugene.models.CNN.configure_sharded_model
      ~eugene.models.CNN.cpu
      ~eugene.models.CNN.cuda
      ~eugene.models.CNN.double
      ~eugene.models.CNN.eval
      ~eugene.models.CNN.extra_repr
      ~eugene.models.CNN.float
      ~eugene.models.CNN.forward
      ~eugene.models.CNN.freeze
      ~eugene.models.CNN.get_buffer
      ~eugene.models.CNN.get_extra_state
      ~eugene.models.CNN.get_from_queue
      ~eugene.models.CNN.get_parameter
      ~eugene.models.CNN.get_progress_bar_dict
      ~eugene.models.CNN.get_submodule
      ~eugene.models.CNN.half
      ~eugene.models.CNN.load_from_checkpoint
      ~eugene.models.CNN.load_state_dict
      ~eugene.models.CNN.log
      ~eugene.models.CNN.log_dict
      ~eugene.models.CNN.log_grad_norm
      ~eugene.models.CNN.lr_schedulers
      ~eugene.models.CNN.manual_backward
      ~eugene.models.CNN.modules
      ~eugene.models.CNN.named_buffers
      ~eugene.models.CNN.named_children
      ~eugene.models.CNN.named_modules
      ~eugene.models.CNN.named_parameters
      ~eugene.models.CNN.on_after_backward
      ~eugene.models.CNN.on_after_batch_transfer
      ~eugene.models.CNN.on_before_backward
      ~eugene.models.CNN.on_before_batch_transfer
      ~eugene.models.CNN.on_before_optimizer_step
      ~eugene.models.CNN.on_before_zero_grad
      ~eugene.models.CNN.on_epoch_end
      ~eugene.models.CNN.on_epoch_start
      ~eugene.models.CNN.on_fit_end
      ~eugene.models.CNN.on_fit_start
      ~eugene.models.CNN.on_hpc_load
      ~eugene.models.CNN.on_hpc_save
      ~eugene.models.CNN.on_load_checkpoint
      ~eugene.models.CNN.on_post_move_to_device
      ~eugene.models.CNN.on_predict_batch_end
      ~eugene.models.CNN.on_predict_batch_start
      ~eugene.models.CNN.on_predict_dataloader
      ~eugene.models.CNN.on_predict_end
      ~eugene.models.CNN.on_predict_epoch_end
      ~eugene.models.CNN.on_predict_epoch_start
      ~eugene.models.CNN.on_predict_model_eval
      ~eugene.models.CNN.on_predict_start
      ~eugene.models.CNN.on_pretrain_routine_end
      ~eugene.models.CNN.on_pretrain_routine_start
      ~eugene.models.CNN.on_save_checkpoint
      ~eugene.models.CNN.on_test_batch_end
      ~eugene.models.CNN.on_test_batch_start
      ~eugene.models.CNN.on_test_dataloader
      ~eugene.models.CNN.on_test_end
      ~eugene.models.CNN.on_test_epoch_end
      ~eugene.models.CNN.on_test_epoch_start
      ~eugene.models.CNN.on_test_model_eval
      ~eugene.models.CNN.on_test_model_train
      ~eugene.models.CNN.on_test_start
      ~eugene.models.CNN.on_train_batch_end
      ~eugene.models.CNN.on_train_batch_start
      ~eugene.models.CNN.on_train_dataloader
      ~eugene.models.CNN.on_train_end
      ~eugene.models.CNN.on_train_epoch_end
      ~eugene.models.CNN.on_train_epoch_start
      ~eugene.models.CNN.on_train_start
      ~eugene.models.CNN.on_val_dataloader
      ~eugene.models.CNN.on_validation_batch_end
      ~eugene.models.CNN.on_validation_batch_start
      ~eugene.models.CNN.on_validation_end
      ~eugene.models.CNN.on_validation_epoch_end
      ~eugene.models.CNN.on_validation_epoch_start
      ~eugene.models.CNN.on_validation_model_eval
      ~eugene.models.CNN.on_validation_model_train
      ~eugene.models.CNN.on_validation_start
      ~eugene.models.CNN.optimizer_step
      ~eugene.models.CNN.optimizer_zero_grad
      ~eugene.models.CNN.optimizers
      ~eugene.models.CNN.parameters
      ~eugene.models.CNN.predict_dataloader
      ~eugene.models.CNN.predict_step
      ~eugene.models.CNN.prepare_data
      ~eugene.models.CNN.print
      ~eugene.models.CNN.register_backward_hook
      ~eugene.models.CNN.register_buffer
      ~eugene.models.CNN.register_forward_hook
      ~eugene.models.CNN.register_forward_pre_hook
      ~eugene.models.CNN.register_full_backward_hook
      ~eugene.models.CNN.register_module
      ~eugene.models.CNN.register_parameter
      ~eugene.models.CNN.requires_grad_
      ~eugene.models.CNN.save_hyperparameters
      ~eugene.models.CNN.set_extra_state
      ~eugene.models.CNN.setup
      ~eugene.models.CNN.share_memory
      ~eugene.models.CNN.state_dict
      ~eugene.models.CNN.summarize
      ~eugene.models.CNN.summary
      ~eugene.models.CNN.tbptt_split_batch
      ~eugene.models.CNN.teardown
      ~eugene.models.CNN.test_dataloader
      ~eugene.models.CNN.test_epoch_end
      ~eugene.models.CNN.test_step
      ~eugene.models.CNN.test_step_end
      ~eugene.models.CNN.to
      ~eugene.models.CNN.to_empty
      ~eugene.models.CNN.to_onnx
      ~eugene.models.CNN.to_torchscript
      ~eugene.models.CNN.toggle_optimizer
      ~eugene.models.CNN.train
      ~eugene.models.CNN.train_dataloader
      ~eugene.models.CNN.training_epoch_end
      ~eugene.models.CNN.training_step
      ~eugene.models.CNN.training_step_end
      ~eugene.models.CNN.transfer_batch_to_device
      ~eugene.models.CNN.type
      ~eugene.models.CNN.unfreeze
      ~eugene.models.CNN.untoggle_optimizer
      ~eugene.models.CNN.val_dataloader
      ~eugene.models.CNN.validation_epoch_end
      ~eugene.models.CNN.validation_step
      ~eugene.models.CNN.validation_step_end
      ~eugene.models.CNN.xpu
      ~eugene.models.CNN.zero_grad



.. _sphx_glr_backref_eugene.models.CNN:
