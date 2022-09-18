eugene.models.FCN
=================

.. currentmodule:: eugene.models

.. autoclass:: FCN


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~FCN.__init__
      ~FCN.add_module
      ~FCN.add_to_queue
      ~FCN.all_gather
      ~FCN.apply
      ~FCN.backward
      ~FCN.bfloat16
      ~FCN.buffers
      ~FCN.children
      ~FCN.clip_gradients
      ~FCN.configure_callbacks
      ~FCN.configure_gradient_clipping
      ~FCN.configure_optimizers
      ~FCN.configure_sharded_model
      ~FCN.cpu
      ~FCN.cuda
      ~FCN.double
      ~FCN.eval
      ~FCN.extra_repr
      ~FCN.float
      ~FCN.forward
      ~FCN.freeze
      ~FCN.get_buffer
      ~FCN.get_extra_state
      ~FCN.get_from_queue
      ~FCN.get_parameter
      ~FCN.get_progress_bar_dict
      ~FCN.get_submodule
      ~FCN.half
      ~FCN.load_from_checkpoint
      ~FCN.load_state_dict
      ~FCN.log
      ~FCN.log_dict
      ~FCN.log_grad_norm
      ~FCN.lr_schedulers
      ~FCN.manual_backward
      ~FCN.modules
      ~FCN.named_buffers
      ~FCN.named_children
      ~FCN.named_modules
      ~FCN.named_parameters
      ~FCN.on_after_backward
      ~FCN.on_after_batch_transfer
      ~FCN.on_before_backward
      ~FCN.on_before_batch_transfer
      ~FCN.on_before_optimizer_step
      ~FCN.on_before_zero_grad
      ~FCN.on_epoch_end
      ~FCN.on_epoch_start
      ~FCN.on_fit_end
      ~FCN.on_fit_start
      ~FCN.on_hpc_load
      ~FCN.on_hpc_save
      ~FCN.on_load_checkpoint
      ~FCN.on_post_move_to_device
      ~FCN.on_predict_batch_end
      ~FCN.on_predict_batch_start
      ~FCN.on_predict_dataloader
      ~FCN.on_predict_end
      ~FCN.on_predict_epoch_end
      ~FCN.on_predict_epoch_start
      ~FCN.on_predict_model_eval
      ~FCN.on_predict_start
      ~FCN.on_pretrain_routine_end
      ~FCN.on_pretrain_routine_start
      ~FCN.on_save_checkpoint
      ~FCN.on_test_batch_end
      ~FCN.on_test_batch_start
      ~FCN.on_test_dataloader
      ~FCN.on_test_end
      ~FCN.on_test_epoch_end
      ~FCN.on_test_epoch_start
      ~FCN.on_test_model_eval
      ~FCN.on_test_model_train
      ~FCN.on_test_start
      ~FCN.on_train_batch_end
      ~FCN.on_train_batch_start
      ~FCN.on_train_dataloader
      ~FCN.on_train_end
      ~FCN.on_train_epoch_end
      ~FCN.on_train_epoch_start
      ~FCN.on_train_start
      ~FCN.on_val_dataloader
      ~FCN.on_validation_batch_end
      ~FCN.on_validation_batch_start
      ~FCN.on_validation_end
      ~FCN.on_validation_epoch_end
      ~FCN.on_validation_epoch_start
      ~FCN.on_validation_model_eval
      ~FCN.on_validation_model_train
      ~FCN.on_validation_start
      ~FCN.optimizer_step
      ~FCN.optimizer_zero_grad
      ~FCN.optimizers
      ~FCN.parameters
      ~FCN.predict_dataloader
      ~FCN.predict_step
      ~FCN.prepare_data
      ~FCN.print
      ~FCN.register_backward_hook
      ~FCN.register_buffer
      ~FCN.register_forward_hook
      ~FCN.register_forward_pre_hook
      ~FCN.register_full_backward_hook
      ~FCN.register_module
      ~FCN.register_parameter
      ~FCN.requires_grad_
      ~FCN.save_hyperparameters
      ~FCN.set_extra_state
      ~FCN.setup
      ~FCN.share_memory
      ~FCN.state_dict
      ~FCN.summarize
      ~FCN.summary
      ~FCN.tbptt_split_batch
      ~FCN.teardown
      ~FCN.test_dataloader
      ~FCN.test_epoch_end
      ~FCN.test_step
      ~FCN.test_step_end
      ~FCN.to
      ~FCN.to_empty
      ~FCN.to_onnx
      ~FCN.to_torchscript
      ~FCN.toggle_optimizer
      ~FCN.train
      ~FCN.train_dataloader
      ~FCN.training_epoch_end
      ~FCN.training_step
      ~FCN.training_step_end
      ~FCN.transfer_batch_to_device
      ~FCN.type
      ~FCN.unfreeze
      ~FCN.untoggle_optimizer
      ~FCN.val_dataloader
      ~FCN.validation_epoch_end
      ~FCN.validation_step
      ~FCN.validation_step_end
      ~FCN.xpu
      ~FCN.zero_grad





   .. rubric:: Attributes

   .. autosummary::

      ~FCN.CHECKPOINT_HYPER_PARAMS_KEY
      ~FCN.CHECKPOINT_HYPER_PARAMS_NAME
      ~FCN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~FCN.T_destination
      ~FCN.automatic_optimization
      ~FCN.current_epoch
      ~FCN.device
      ~FCN.dtype
      ~FCN.dump_patches
      ~FCN.example_input_array
      ~FCN.global_rank
      ~FCN.global_step
      ~FCN.hparams
      ~FCN.hparams_initial
      ~FCN.loaded_optimizer_states_dict
      ~FCN.local_rank
      ~FCN.logger
      ~FCN.model_size
      ~FCN.on_gpu
      ~FCN.truncated_bptt_steps
      ~FCN.training
