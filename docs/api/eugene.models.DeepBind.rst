eugene.models.DeepBind
======================

.. currentmodule:: eugene.models

.. autoclass:: DeepBind


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~DeepBind.__init__
      ~DeepBind.add_module
      ~DeepBind.add_to_queue
      ~DeepBind.all_gather
      ~DeepBind.apply
      ~DeepBind.backward
      ~DeepBind.bfloat16
      ~DeepBind.buffers
      ~DeepBind.children
      ~DeepBind.clip_gradients
      ~DeepBind.configure_callbacks
      ~DeepBind.configure_gradient_clipping
      ~DeepBind.configure_optimizers
      ~DeepBind.configure_sharded_model
      ~DeepBind.cpu
      ~DeepBind.cuda
      ~DeepBind.double
      ~DeepBind.eval
      ~DeepBind.extra_repr
      ~DeepBind.float
      ~DeepBind.forward
      ~DeepBind.freeze
      ~DeepBind.get_buffer
      ~DeepBind.get_extra_state
      ~DeepBind.get_from_queue
      ~DeepBind.get_parameter
      ~DeepBind.get_progress_bar_dict
      ~DeepBind.get_submodule
      ~DeepBind.half
      ~DeepBind.kwarg_handler
      ~DeepBind.load_from_checkpoint
      ~DeepBind.load_state_dict
      ~DeepBind.log
      ~DeepBind.log_dict
      ~DeepBind.log_grad_norm
      ~DeepBind.lr_schedulers
      ~DeepBind.manual_backward
      ~DeepBind.modules
      ~DeepBind.named_buffers
      ~DeepBind.named_children
      ~DeepBind.named_modules
      ~DeepBind.named_parameters
      ~DeepBind.on_after_backward
      ~DeepBind.on_after_batch_transfer
      ~DeepBind.on_before_backward
      ~DeepBind.on_before_batch_transfer
      ~DeepBind.on_before_optimizer_step
      ~DeepBind.on_before_zero_grad
      ~DeepBind.on_epoch_end
      ~DeepBind.on_epoch_start
      ~DeepBind.on_fit_end
      ~DeepBind.on_fit_start
      ~DeepBind.on_hpc_load
      ~DeepBind.on_hpc_save
      ~DeepBind.on_load_checkpoint
      ~DeepBind.on_post_move_to_device
      ~DeepBind.on_predict_batch_end
      ~DeepBind.on_predict_batch_start
      ~DeepBind.on_predict_dataloader
      ~DeepBind.on_predict_end
      ~DeepBind.on_predict_epoch_end
      ~DeepBind.on_predict_epoch_start
      ~DeepBind.on_predict_model_eval
      ~DeepBind.on_predict_start
      ~DeepBind.on_pretrain_routine_end
      ~DeepBind.on_pretrain_routine_start
      ~DeepBind.on_save_checkpoint
      ~DeepBind.on_test_batch_end
      ~DeepBind.on_test_batch_start
      ~DeepBind.on_test_dataloader
      ~DeepBind.on_test_end
      ~DeepBind.on_test_epoch_end
      ~DeepBind.on_test_epoch_start
      ~DeepBind.on_test_model_eval
      ~DeepBind.on_test_model_train
      ~DeepBind.on_test_start
      ~DeepBind.on_train_batch_end
      ~DeepBind.on_train_batch_start
      ~DeepBind.on_train_dataloader
      ~DeepBind.on_train_end
      ~DeepBind.on_train_epoch_end
      ~DeepBind.on_train_epoch_start
      ~DeepBind.on_train_start
      ~DeepBind.on_val_dataloader
      ~DeepBind.on_validation_batch_end
      ~DeepBind.on_validation_batch_start
      ~DeepBind.on_validation_end
      ~DeepBind.on_validation_epoch_end
      ~DeepBind.on_validation_epoch_start
      ~DeepBind.on_validation_model_eval
      ~DeepBind.on_validation_model_train
      ~DeepBind.on_validation_start
      ~DeepBind.optimizer_step
      ~DeepBind.optimizer_zero_grad
      ~DeepBind.optimizers
      ~DeepBind.parameters
      ~DeepBind.predict_dataloader
      ~DeepBind.predict_step
      ~DeepBind.prepare_data
      ~DeepBind.print
      ~DeepBind.register_backward_hook
      ~DeepBind.register_buffer
      ~DeepBind.register_forward_hook
      ~DeepBind.register_forward_pre_hook
      ~DeepBind.register_full_backward_hook
      ~DeepBind.register_module
      ~DeepBind.register_parameter
      ~DeepBind.requires_grad_
      ~DeepBind.save_hyperparameters
      ~DeepBind.set_extra_state
      ~DeepBind.setup
      ~DeepBind.share_memory
      ~DeepBind.state_dict
      ~DeepBind.summarize
      ~DeepBind.summary
      ~DeepBind.tbptt_split_batch
      ~DeepBind.teardown
      ~DeepBind.test_dataloader
      ~DeepBind.test_epoch_end
      ~DeepBind.test_step
      ~DeepBind.test_step_end
      ~DeepBind.to
      ~DeepBind.to_empty
      ~DeepBind.to_onnx
      ~DeepBind.to_torchscript
      ~DeepBind.toggle_optimizer
      ~DeepBind.train
      ~DeepBind.train_dataloader
      ~DeepBind.training_epoch_end
      ~DeepBind.training_step
      ~DeepBind.training_step_end
      ~DeepBind.transfer_batch_to_device
      ~DeepBind.type
      ~DeepBind.unfreeze
      ~DeepBind.untoggle_optimizer
      ~DeepBind.val_dataloader
      ~DeepBind.validation_epoch_end
      ~DeepBind.validation_step
      ~DeepBind.validation_step_end
      ~DeepBind.xpu
      ~DeepBind.zero_grad





   .. rubric:: Attributes

   .. autosummary::

      ~DeepBind.CHECKPOINT_HYPER_PARAMS_KEY
      ~DeepBind.CHECKPOINT_HYPER_PARAMS_NAME
      ~DeepBind.CHECKPOINT_HYPER_PARAMS_TYPE
      ~DeepBind.T_destination
      ~DeepBind.automatic_optimization
      ~DeepBind.current_epoch
      ~DeepBind.device
      ~DeepBind.dtype
      ~DeepBind.dump_patches
      ~DeepBind.example_input_array
      ~DeepBind.global_rank
      ~DeepBind.global_step
      ~DeepBind.hparams
      ~DeepBind.hparams_initial
      ~DeepBind.loaded_optimizer_states_dict
      ~DeepBind.local_rank
      ~DeepBind.logger
      ~DeepBind.model_size
      ~DeepBind.on_gpu
      ~DeepBind.truncated_bptt_steps
      ~DeepBind.training
