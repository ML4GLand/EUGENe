eugene.models.DeepSEA
=====================

.. currentmodule:: eugene.models

.. autoclass:: DeepSEA


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~DeepSEA.__init__
      ~DeepSEA.add_module
      ~DeepSEA.add_to_queue
      ~DeepSEA.all_gather
      ~DeepSEA.apply
      ~DeepSEA.backward
      ~DeepSEA.bfloat16
      ~DeepSEA.buffers
      ~DeepSEA.children
      ~DeepSEA.clip_gradients
      ~DeepSEA.configure_callbacks
      ~DeepSEA.configure_gradient_clipping
      ~DeepSEA.configure_optimizers
      ~DeepSEA.configure_sharded_model
      ~DeepSEA.cpu
      ~DeepSEA.cuda
      ~DeepSEA.double
      ~DeepSEA.eval
      ~DeepSEA.extra_repr
      ~DeepSEA.float
      ~DeepSEA.forward
      ~DeepSEA.freeze
      ~DeepSEA.get_buffer
      ~DeepSEA.get_extra_state
      ~DeepSEA.get_from_queue
      ~DeepSEA.get_parameter
      ~DeepSEA.get_progress_bar_dict
      ~DeepSEA.get_submodule
      ~DeepSEA.half
      ~DeepSEA.kwarg_handler
      ~DeepSEA.load_from_checkpoint
      ~DeepSEA.load_state_dict
      ~DeepSEA.log
      ~DeepSEA.log_dict
      ~DeepSEA.log_grad_norm
      ~DeepSEA.lr_schedulers
      ~DeepSEA.manual_backward
      ~DeepSEA.modules
      ~DeepSEA.named_buffers
      ~DeepSEA.named_children
      ~DeepSEA.named_modules
      ~DeepSEA.named_parameters
      ~DeepSEA.on_after_backward
      ~DeepSEA.on_after_batch_transfer
      ~DeepSEA.on_before_backward
      ~DeepSEA.on_before_batch_transfer
      ~DeepSEA.on_before_optimizer_step
      ~DeepSEA.on_before_zero_grad
      ~DeepSEA.on_epoch_end
      ~DeepSEA.on_epoch_start
      ~DeepSEA.on_fit_end
      ~DeepSEA.on_fit_start
      ~DeepSEA.on_hpc_load
      ~DeepSEA.on_hpc_save
      ~DeepSEA.on_load_checkpoint
      ~DeepSEA.on_post_move_to_device
      ~DeepSEA.on_predict_batch_end
      ~DeepSEA.on_predict_batch_start
      ~DeepSEA.on_predict_dataloader
      ~DeepSEA.on_predict_end
      ~DeepSEA.on_predict_epoch_end
      ~DeepSEA.on_predict_epoch_start
      ~DeepSEA.on_predict_model_eval
      ~DeepSEA.on_predict_start
      ~DeepSEA.on_pretrain_routine_end
      ~DeepSEA.on_pretrain_routine_start
      ~DeepSEA.on_save_checkpoint
      ~DeepSEA.on_test_batch_end
      ~DeepSEA.on_test_batch_start
      ~DeepSEA.on_test_dataloader
      ~DeepSEA.on_test_end
      ~DeepSEA.on_test_epoch_end
      ~DeepSEA.on_test_epoch_start
      ~DeepSEA.on_test_model_eval
      ~DeepSEA.on_test_model_train
      ~DeepSEA.on_test_start
      ~DeepSEA.on_train_batch_end
      ~DeepSEA.on_train_batch_start
      ~DeepSEA.on_train_dataloader
      ~DeepSEA.on_train_end
      ~DeepSEA.on_train_epoch_end
      ~DeepSEA.on_train_epoch_start
      ~DeepSEA.on_train_start
      ~DeepSEA.on_val_dataloader
      ~DeepSEA.on_validation_batch_end
      ~DeepSEA.on_validation_batch_start
      ~DeepSEA.on_validation_end
      ~DeepSEA.on_validation_epoch_end
      ~DeepSEA.on_validation_epoch_start
      ~DeepSEA.on_validation_model_eval
      ~DeepSEA.on_validation_model_train
      ~DeepSEA.on_validation_start
      ~DeepSEA.optimizer_step
      ~DeepSEA.optimizer_zero_grad
      ~DeepSEA.optimizers
      ~DeepSEA.parameters
      ~DeepSEA.predict_dataloader
      ~DeepSEA.predict_step
      ~DeepSEA.prepare_data
      ~DeepSEA.print
      ~DeepSEA.register_backward_hook
      ~DeepSEA.register_buffer
      ~DeepSEA.register_forward_hook
      ~DeepSEA.register_forward_pre_hook
      ~DeepSEA.register_full_backward_hook
      ~DeepSEA.register_module
      ~DeepSEA.register_parameter
      ~DeepSEA.requires_grad_
      ~DeepSEA.save_hyperparameters
      ~DeepSEA.set_extra_state
      ~DeepSEA.setup
      ~DeepSEA.share_memory
      ~DeepSEA.state_dict
      ~DeepSEA.summarize
      ~DeepSEA.summary
      ~DeepSEA.tbptt_split_batch
      ~DeepSEA.teardown
      ~DeepSEA.test_dataloader
      ~DeepSEA.test_epoch_end
      ~DeepSEA.test_step
      ~DeepSEA.test_step_end
      ~DeepSEA.to
      ~DeepSEA.to_empty
      ~DeepSEA.to_onnx
      ~DeepSEA.to_torchscript
      ~DeepSEA.toggle_optimizer
      ~DeepSEA.train
      ~DeepSEA.train_dataloader
      ~DeepSEA.training_epoch_end
      ~DeepSEA.training_step
      ~DeepSEA.training_step_end
      ~DeepSEA.transfer_batch_to_device
      ~DeepSEA.type
      ~DeepSEA.unfreeze
      ~DeepSEA.untoggle_optimizer
      ~DeepSEA.val_dataloader
      ~DeepSEA.validation_epoch_end
      ~DeepSEA.validation_step
      ~DeepSEA.validation_step_end
      ~DeepSEA.xpu
      ~DeepSEA.zero_grad





   .. rubric:: Attributes

   .. autosummary::

      ~DeepSEA.CHECKPOINT_HYPER_PARAMS_KEY
      ~DeepSEA.CHECKPOINT_HYPER_PARAMS_NAME
      ~DeepSEA.CHECKPOINT_HYPER_PARAMS_TYPE
      ~DeepSEA.T_destination
      ~DeepSEA.automatic_optimization
      ~DeepSEA.current_epoch
      ~DeepSEA.device
      ~DeepSEA.dtype
      ~DeepSEA.dump_patches
      ~DeepSEA.example_input_array
      ~DeepSEA.global_rank
      ~DeepSEA.global_step
      ~DeepSEA.hparams
      ~DeepSEA.hparams_initial
      ~DeepSEA.loaded_optimizer_states_dict
      ~DeepSEA.local_rank
      ~DeepSEA.logger
      ~DeepSEA.model_size
      ~DeepSEA.on_gpu
      ~DeepSEA.truncated_bptt_steps
      ~DeepSEA.training
