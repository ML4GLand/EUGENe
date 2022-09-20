:github_url: eugene.models.FCN

eugene.models.FCN
=================

.. currentmodule:: eugene.models

.. autoclass:: FCN



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.FCN.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.FCN.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.FCN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.FCN.T_destination
      ~eugene.models.FCN.automatic_optimization
      ~eugene.models.FCN.current_epoch
      ~eugene.models.FCN.device
      ~eugene.models.FCN.dtype
      ~eugene.models.FCN.dump_patches
      ~eugene.models.FCN.example_input_array
      ~eugene.models.FCN.global_rank
      ~eugene.models.FCN.global_step
      ~eugene.models.FCN.hparams
      ~eugene.models.FCN.hparams_initial
      ~eugene.models.FCN.loaded_optimizer_states_dict
      ~eugene.models.FCN.local_rank
      ~eugene.models.FCN.logger
      ~eugene.models.FCN.model_size
      ~eugene.models.FCN.on_gpu
      ~eugene.models.FCN.truncated_bptt_steps
      ~eugene.models.FCN.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.FCN.add_module
      ~eugene.models.FCN.add_to_queue
      ~eugene.models.FCN.all_gather
      ~eugene.models.FCN.apply
      ~eugene.models.FCN.backward
      ~eugene.models.FCN.bfloat16
      ~eugene.models.FCN.buffers
      ~eugene.models.FCN.children
      ~eugene.models.FCN.clip_gradients
      ~eugene.models.FCN.configure_callbacks
      ~eugene.models.FCN.configure_gradient_clipping
      ~eugene.models.FCN.configure_optimizers
      ~eugene.models.FCN.configure_sharded_model
      ~eugene.models.FCN.cpu
      ~eugene.models.FCN.cuda
      ~eugene.models.FCN.double
      ~eugene.models.FCN.eval
      ~eugene.models.FCN.extra_repr
      ~eugene.models.FCN.float
      ~eugene.models.FCN.forward
      ~eugene.models.FCN.freeze
      ~eugene.models.FCN.get_buffer
      ~eugene.models.FCN.get_extra_state
      ~eugene.models.FCN.get_from_queue
      ~eugene.models.FCN.get_parameter
      ~eugene.models.FCN.get_progress_bar_dict
      ~eugene.models.FCN.get_submodule
      ~eugene.models.FCN.half
      ~eugene.models.FCN.load_from_checkpoint
      ~eugene.models.FCN.load_state_dict
      ~eugene.models.FCN.log
      ~eugene.models.FCN.log_dict
      ~eugene.models.FCN.log_grad_norm
      ~eugene.models.FCN.lr_schedulers
      ~eugene.models.FCN.manual_backward
      ~eugene.models.FCN.modules
      ~eugene.models.FCN.named_buffers
      ~eugene.models.FCN.named_children
      ~eugene.models.FCN.named_modules
      ~eugene.models.FCN.named_parameters
      ~eugene.models.FCN.on_after_backward
      ~eugene.models.FCN.on_after_batch_transfer
      ~eugene.models.FCN.on_before_backward
      ~eugene.models.FCN.on_before_batch_transfer
      ~eugene.models.FCN.on_before_optimizer_step
      ~eugene.models.FCN.on_before_zero_grad
      ~eugene.models.FCN.on_epoch_end
      ~eugene.models.FCN.on_epoch_start
      ~eugene.models.FCN.on_fit_end
      ~eugene.models.FCN.on_fit_start
      ~eugene.models.FCN.on_hpc_load
      ~eugene.models.FCN.on_hpc_save
      ~eugene.models.FCN.on_load_checkpoint
      ~eugene.models.FCN.on_post_move_to_device
      ~eugene.models.FCN.on_predict_batch_end
      ~eugene.models.FCN.on_predict_batch_start
      ~eugene.models.FCN.on_predict_dataloader
      ~eugene.models.FCN.on_predict_end
      ~eugene.models.FCN.on_predict_epoch_end
      ~eugene.models.FCN.on_predict_epoch_start
      ~eugene.models.FCN.on_predict_model_eval
      ~eugene.models.FCN.on_predict_start
      ~eugene.models.FCN.on_pretrain_routine_end
      ~eugene.models.FCN.on_pretrain_routine_start
      ~eugene.models.FCN.on_save_checkpoint
      ~eugene.models.FCN.on_test_batch_end
      ~eugene.models.FCN.on_test_batch_start
      ~eugene.models.FCN.on_test_dataloader
      ~eugene.models.FCN.on_test_end
      ~eugene.models.FCN.on_test_epoch_end
      ~eugene.models.FCN.on_test_epoch_start
      ~eugene.models.FCN.on_test_model_eval
      ~eugene.models.FCN.on_test_model_train
      ~eugene.models.FCN.on_test_start
      ~eugene.models.FCN.on_train_batch_end
      ~eugene.models.FCN.on_train_batch_start
      ~eugene.models.FCN.on_train_dataloader
      ~eugene.models.FCN.on_train_end
      ~eugene.models.FCN.on_train_epoch_end
      ~eugene.models.FCN.on_train_epoch_start
      ~eugene.models.FCN.on_train_start
      ~eugene.models.FCN.on_val_dataloader
      ~eugene.models.FCN.on_validation_batch_end
      ~eugene.models.FCN.on_validation_batch_start
      ~eugene.models.FCN.on_validation_end
      ~eugene.models.FCN.on_validation_epoch_end
      ~eugene.models.FCN.on_validation_epoch_start
      ~eugene.models.FCN.on_validation_model_eval
      ~eugene.models.FCN.on_validation_model_train
      ~eugene.models.FCN.on_validation_start
      ~eugene.models.FCN.optimizer_step
      ~eugene.models.FCN.optimizer_zero_grad
      ~eugene.models.FCN.optimizers
      ~eugene.models.FCN.parameters
      ~eugene.models.FCN.predict_dataloader
      ~eugene.models.FCN.predict_step
      ~eugene.models.FCN.prepare_data
      ~eugene.models.FCN.print
      ~eugene.models.FCN.register_backward_hook
      ~eugene.models.FCN.register_buffer
      ~eugene.models.FCN.register_forward_hook
      ~eugene.models.FCN.register_forward_pre_hook
      ~eugene.models.FCN.register_full_backward_hook
      ~eugene.models.FCN.register_module
      ~eugene.models.FCN.register_parameter
      ~eugene.models.FCN.requires_grad_
      ~eugene.models.FCN.save_hyperparameters
      ~eugene.models.FCN.set_extra_state
      ~eugene.models.FCN.setup
      ~eugene.models.FCN.share_memory
      ~eugene.models.FCN.state_dict
      ~eugene.models.FCN.summarize
      ~eugene.models.FCN.summary
      ~eugene.models.FCN.tbptt_split_batch
      ~eugene.models.FCN.teardown
      ~eugene.models.FCN.test_dataloader
      ~eugene.models.FCN.test_epoch_end
      ~eugene.models.FCN.test_step
      ~eugene.models.FCN.test_step_end
      ~eugene.models.FCN.to
      ~eugene.models.FCN.to_empty
      ~eugene.models.FCN.to_onnx
      ~eugene.models.FCN.to_torchscript
      ~eugene.models.FCN.toggle_optimizer
      ~eugene.models.FCN.train
      ~eugene.models.FCN.train_dataloader
      ~eugene.models.FCN.training_epoch_end
      ~eugene.models.FCN.training_step
      ~eugene.models.FCN.training_step_end
      ~eugene.models.FCN.transfer_batch_to_device
      ~eugene.models.FCN.type
      ~eugene.models.FCN.unfreeze
      ~eugene.models.FCN.untoggle_optimizer
      ~eugene.models.FCN.val_dataloader
      ~eugene.models.FCN.validation_epoch_end
      ~eugene.models.FCN.validation_step
      ~eugene.models.FCN.validation_step_end
      ~eugene.models.FCN.xpu
      ~eugene.models.FCN.zero_grad



.. _sphx_glr_backref_eugene.models.FCN:
