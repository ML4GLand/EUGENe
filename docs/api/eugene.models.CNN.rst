eugene.models.CNN
=================

.. currentmodule:: eugene.models

.. autoclass:: CNN


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~CNN.__init__
      ~CNN.add_module
      ~CNN.add_to_queue
      ~CNN.all_gather
      ~CNN.apply
      ~CNN.backward
      ~CNN.bfloat16
      ~CNN.buffers
      ~CNN.children
      ~CNN.clip_gradients
      ~CNN.configure_callbacks
      ~CNN.configure_gradient_clipping
      ~CNN.configure_optimizers
      ~CNN.configure_sharded_model
      ~CNN.cpu
      ~CNN.cuda
      ~CNN.double
      ~CNN.eval
      ~CNN.extra_repr
      ~CNN.float
      ~CNN.forward
      ~CNN.freeze
      ~CNN.get_buffer
      ~CNN.get_extra_state
      ~CNN.get_from_queue
      ~CNN.get_parameter
      ~CNN.get_progress_bar_dict
      ~CNN.get_submodule
      ~CNN.half
      ~CNN.load_from_checkpoint
      ~CNN.load_state_dict
      ~CNN.log
      ~CNN.log_dict
      ~CNN.log_grad_norm
      ~CNN.lr_schedulers
      ~CNN.manual_backward
      ~CNN.modules
      ~CNN.named_buffers
      ~CNN.named_children
      ~CNN.named_modules
      ~CNN.named_parameters
      ~CNN.on_after_backward
      ~CNN.on_after_batch_transfer
      ~CNN.on_before_backward
      ~CNN.on_before_batch_transfer
      ~CNN.on_before_optimizer_step
      ~CNN.on_before_zero_grad
      ~CNN.on_epoch_end
      ~CNN.on_epoch_start
      ~CNN.on_fit_end
      ~CNN.on_fit_start
      ~CNN.on_hpc_load
      ~CNN.on_hpc_save
      ~CNN.on_load_checkpoint
      ~CNN.on_post_move_to_device
      ~CNN.on_predict_batch_end
      ~CNN.on_predict_batch_start
      ~CNN.on_predict_dataloader
      ~CNN.on_predict_end
      ~CNN.on_predict_epoch_end
      ~CNN.on_predict_epoch_start
      ~CNN.on_predict_model_eval
      ~CNN.on_predict_start
      ~CNN.on_pretrain_routine_end
      ~CNN.on_pretrain_routine_start
      ~CNN.on_save_checkpoint
      ~CNN.on_test_batch_end
      ~CNN.on_test_batch_start
      ~CNN.on_test_dataloader
      ~CNN.on_test_end
      ~CNN.on_test_epoch_end
      ~CNN.on_test_epoch_start
      ~CNN.on_test_model_eval
      ~CNN.on_test_model_train
      ~CNN.on_test_start
      ~CNN.on_train_batch_end
      ~CNN.on_train_batch_start
      ~CNN.on_train_dataloader
      ~CNN.on_train_end
      ~CNN.on_train_epoch_end
      ~CNN.on_train_epoch_start
      ~CNN.on_train_start
      ~CNN.on_val_dataloader
      ~CNN.on_validation_batch_end
      ~CNN.on_validation_batch_start
      ~CNN.on_validation_end
      ~CNN.on_validation_epoch_end
      ~CNN.on_validation_epoch_start
      ~CNN.on_validation_model_eval
      ~CNN.on_validation_model_train
      ~CNN.on_validation_start
      ~CNN.optimizer_step
      ~CNN.optimizer_zero_grad
      ~CNN.optimizers
      ~CNN.parameters
      ~CNN.predict_dataloader
      ~CNN.predict_step
      ~CNN.prepare_data
      ~CNN.print
      ~CNN.register_backward_hook
      ~CNN.register_buffer
      ~CNN.register_forward_hook
      ~CNN.register_forward_pre_hook
      ~CNN.register_full_backward_hook
      ~CNN.register_module
      ~CNN.register_parameter
      ~CNN.requires_grad_
      ~CNN.save_hyperparameters
      ~CNN.set_extra_state
      ~CNN.setup
      ~CNN.share_memory
      ~CNN.state_dict
      ~CNN.summarize
      ~CNN.summary
      ~CNN.tbptt_split_batch
      ~CNN.teardown
      ~CNN.test_dataloader
      ~CNN.test_epoch_end
      ~CNN.test_step
      ~CNN.test_step_end
      ~CNN.to
      ~CNN.to_empty
      ~CNN.to_onnx
      ~CNN.to_torchscript
      ~CNN.toggle_optimizer
      ~CNN.train
      ~CNN.train_dataloader
      ~CNN.training_epoch_end
      ~CNN.training_step
      ~CNN.training_step_end
      ~CNN.transfer_batch_to_device
      ~CNN.type
      ~CNN.unfreeze
      ~CNN.untoggle_optimizer
      ~CNN.val_dataloader
      ~CNN.validation_epoch_end
      ~CNN.validation_step
      ~CNN.validation_step_end
      ~CNN.xpu
      ~CNN.zero_grad





   .. rubric:: Attributes

   .. autosummary::

      ~CNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~CNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~CNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~CNN.T_destination
      ~CNN.automatic_optimization
      ~CNN.current_epoch
      ~CNN.device
      ~CNN.dtype
      ~CNN.dump_patches
      ~CNN.example_input_array
      ~CNN.global_rank
      ~CNN.global_step
      ~CNN.hparams
      ~CNN.hparams_initial
      ~CNN.loaded_optimizer_states_dict
      ~CNN.local_rank
      ~CNN.logger
      ~CNN.model_size
      ~CNN.on_gpu
      ~CNN.truncated_bptt_steps
      ~CNN.training
