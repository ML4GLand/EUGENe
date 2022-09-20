:github_url: eugene.models.Kopp21CNN

eugene.models.Kopp21CNN
=======================

.. currentmodule:: eugene.models

.. autoclass:: Kopp21CNN



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.Kopp21CNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.Kopp21CNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.Kopp21CNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.Kopp21CNN.T_destination
      ~eugene.models.Kopp21CNN.automatic_optimization
      ~eugene.models.Kopp21CNN.current_epoch
      ~eugene.models.Kopp21CNN.device
      ~eugene.models.Kopp21CNN.dtype
      ~eugene.models.Kopp21CNN.dump_patches
      ~eugene.models.Kopp21CNN.example_input_array
      ~eugene.models.Kopp21CNN.global_rank
      ~eugene.models.Kopp21CNN.global_step
      ~eugene.models.Kopp21CNN.hparams
      ~eugene.models.Kopp21CNN.hparams_initial
      ~eugene.models.Kopp21CNN.loaded_optimizer_states_dict
      ~eugene.models.Kopp21CNN.local_rank
      ~eugene.models.Kopp21CNN.logger
      ~eugene.models.Kopp21CNN.model_size
      ~eugene.models.Kopp21CNN.on_gpu
      ~eugene.models.Kopp21CNN.truncated_bptt_steps
      ~eugene.models.Kopp21CNN.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.Kopp21CNN.add_module
      ~eugene.models.Kopp21CNN.add_to_queue
      ~eugene.models.Kopp21CNN.all_gather
      ~eugene.models.Kopp21CNN.apply
      ~eugene.models.Kopp21CNN.backward
      ~eugene.models.Kopp21CNN.bfloat16
      ~eugene.models.Kopp21CNN.buffers
      ~eugene.models.Kopp21CNN.children
      ~eugene.models.Kopp21CNN.clip_gradients
      ~eugene.models.Kopp21CNN.configure_callbacks
      ~eugene.models.Kopp21CNN.configure_gradient_clipping
      ~eugene.models.Kopp21CNN.configure_optimizers
      ~eugene.models.Kopp21CNN.configure_sharded_model
      ~eugene.models.Kopp21CNN.cpu
      ~eugene.models.Kopp21CNN.cuda
      ~eugene.models.Kopp21CNN.double
      ~eugene.models.Kopp21CNN.eval
      ~eugene.models.Kopp21CNN.extra_repr
      ~eugene.models.Kopp21CNN.float
      ~eugene.models.Kopp21CNN.forward
      ~eugene.models.Kopp21CNN.freeze
      ~eugene.models.Kopp21CNN.get_buffer
      ~eugene.models.Kopp21CNN.get_extra_state
      ~eugene.models.Kopp21CNN.get_from_queue
      ~eugene.models.Kopp21CNN.get_parameter
      ~eugene.models.Kopp21CNN.get_progress_bar_dict
      ~eugene.models.Kopp21CNN.get_submodule
      ~eugene.models.Kopp21CNN.half
      ~eugene.models.Kopp21CNN.load_from_checkpoint
      ~eugene.models.Kopp21CNN.load_state_dict
      ~eugene.models.Kopp21CNN.log
      ~eugene.models.Kopp21CNN.log_dict
      ~eugene.models.Kopp21CNN.log_grad_norm
      ~eugene.models.Kopp21CNN.lr_schedulers
      ~eugene.models.Kopp21CNN.manual_backward
      ~eugene.models.Kopp21CNN.modules
      ~eugene.models.Kopp21CNN.named_buffers
      ~eugene.models.Kopp21CNN.named_children
      ~eugene.models.Kopp21CNN.named_modules
      ~eugene.models.Kopp21CNN.named_parameters
      ~eugene.models.Kopp21CNN.on_after_backward
      ~eugene.models.Kopp21CNN.on_after_batch_transfer
      ~eugene.models.Kopp21CNN.on_before_backward
      ~eugene.models.Kopp21CNN.on_before_batch_transfer
      ~eugene.models.Kopp21CNN.on_before_optimizer_step
      ~eugene.models.Kopp21CNN.on_before_zero_grad
      ~eugene.models.Kopp21CNN.on_epoch_end
      ~eugene.models.Kopp21CNN.on_epoch_start
      ~eugene.models.Kopp21CNN.on_fit_end
      ~eugene.models.Kopp21CNN.on_fit_start
      ~eugene.models.Kopp21CNN.on_hpc_load
      ~eugene.models.Kopp21CNN.on_hpc_save
      ~eugene.models.Kopp21CNN.on_load_checkpoint
      ~eugene.models.Kopp21CNN.on_post_move_to_device
      ~eugene.models.Kopp21CNN.on_predict_batch_end
      ~eugene.models.Kopp21CNN.on_predict_batch_start
      ~eugene.models.Kopp21CNN.on_predict_dataloader
      ~eugene.models.Kopp21CNN.on_predict_end
      ~eugene.models.Kopp21CNN.on_predict_epoch_end
      ~eugene.models.Kopp21CNN.on_predict_epoch_start
      ~eugene.models.Kopp21CNN.on_predict_model_eval
      ~eugene.models.Kopp21CNN.on_predict_start
      ~eugene.models.Kopp21CNN.on_pretrain_routine_end
      ~eugene.models.Kopp21CNN.on_pretrain_routine_start
      ~eugene.models.Kopp21CNN.on_save_checkpoint
      ~eugene.models.Kopp21CNN.on_test_batch_end
      ~eugene.models.Kopp21CNN.on_test_batch_start
      ~eugene.models.Kopp21CNN.on_test_dataloader
      ~eugene.models.Kopp21CNN.on_test_end
      ~eugene.models.Kopp21CNN.on_test_epoch_end
      ~eugene.models.Kopp21CNN.on_test_epoch_start
      ~eugene.models.Kopp21CNN.on_test_model_eval
      ~eugene.models.Kopp21CNN.on_test_model_train
      ~eugene.models.Kopp21CNN.on_test_start
      ~eugene.models.Kopp21CNN.on_train_batch_end
      ~eugene.models.Kopp21CNN.on_train_batch_start
      ~eugene.models.Kopp21CNN.on_train_dataloader
      ~eugene.models.Kopp21CNN.on_train_end
      ~eugene.models.Kopp21CNN.on_train_epoch_end
      ~eugene.models.Kopp21CNN.on_train_epoch_start
      ~eugene.models.Kopp21CNN.on_train_start
      ~eugene.models.Kopp21CNN.on_val_dataloader
      ~eugene.models.Kopp21CNN.on_validation_batch_end
      ~eugene.models.Kopp21CNN.on_validation_batch_start
      ~eugene.models.Kopp21CNN.on_validation_end
      ~eugene.models.Kopp21CNN.on_validation_epoch_end
      ~eugene.models.Kopp21CNN.on_validation_epoch_start
      ~eugene.models.Kopp21CNN.on_validation_model_eval
      ~eugene.models.Kopp21CNN.on_validation_model_train
      ~eugene.models.Kopp21CNN.on_validation_start
      ~eugene.models.Kopp21CNN.optimizer_step
      ~eugene.models.Kopp21CNN.optimizer_zero_grad
      ~eugene.models.Kopp21CNN.optimizers
      ~eugene.models.Kopp21CNN.parameters
      ~eugene.models.Kopp21CNN.predict_dataloader
      ~eugene.models.Kopp21CNN.predict_step
      ~eugene.models.Kopp21CNN.prepare_data
      ~eugene.models.Kopp21CNN.print
      ~eugene.models.Kopp21CNN.register_backward_hook
      ~eugene.models.Kopp21CNN.register_buffer
      ~eugene.models.Kopp21CNN.register_forward_hook
      ~eugene.models.Kopp21CNN.register_forward_pre_hook
      ~eugene.models.Kopp21CNN.register_full_backward_hook
      ~eugene.models.Kopp21CNN.register_module
      ~eugene.models.Kopp21CNN.register_parameter
      ~eugene.models.Kopp21CNN.requires_grad_
      ~eugene.models.Kopp21CNN.save_hyperparameters
      ~eugene.models.Kopp21CNN.set_extra_state
      ~eugene.models.Kopp21CNN.setup
      ~eugene.models.Kopp21CNN.share_memory
      ~eugene.models.Kopp21CNN.state_dict
      ~eugene.models.Kopp21CNN.summarize
      ~eugene.models.Kopp21CNN.summary
      ~eugene.models.Kopp21CNN.tbptt_split_batch
      ~eugene.models.Kopp21CNN.teardown
      ~eugene.models.Kopp21CNN.test_dataloader
      ~eugene.models.Kopp21CNN.test_epoch_end
      ~eugene.models.Kopp21CNN.test_step
      ~eugene.models.Kopp21CNN.test_step_end
      ~eugene.models.Kopp21CNN.to
      ~eugene.models.Kopp21CNN.to_empty
      ~eugene.models.Kopp21CNN.to_onnx
      ~eugene.models.Kopp21CNN.to_torchscript
      ~eugene.models.Kopp21CNN.toggle_optimizer
      ~eugene.models.Kopp21CNN.train
      ~eugene.models.Kopp21CNN.train_dataloader
      ~eugene.models.Kopp21CNN.training_epoch_end
      ~eugene.models.Kopp21CNN.training_step
      ~eugene.models.Kopp21CNN.training_step_end
      ~eugene.models.Kopp21CNN.transfer_batch_to_device
      ~eugene.models.Kopp21CNN.type
      ~eugene.models.Kopp21CNN.unfreeze
      ~eugene.models.Kopp21CNN.untoggle_optimizer
      ~eugene.models.Kopp21CNN.val_dataloader
      ~eugene.models.Kopp21CNN.validation_epoch_end
      ~eugene.models.Kopp21CNN.validation_step
      ~eugene.models.Kopp21CNN.validation_step_end
      ~eugene.models.Kopp21CNN.xpu
      ~eugene.models.Kopp21CNN.zero_grad



.. _sphx_glr_backref_eugene.models.Kopp21CNN:
