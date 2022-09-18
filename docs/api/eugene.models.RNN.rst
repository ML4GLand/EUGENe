eugene.models.RNN
=================

.. currentmodule:: eugene.models

.. autoclass:: RNN


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~RNN.__init__
      ~RNN.add_module
      ~RNN.add_to_queue
      ~RNN.all_gather
      ~RNN.apply
      ~RNN.backward
      ~RNN.bfloat16
      ~RNN.buffers
      ~RNN.children
      ~RNN.clip_gradients
      ~RNN.configure_callbacks
      ~RNN.configure_gradient_clipping
      ~RNN.configure_optimizers
      ~RNN.configure_sharded_model
      ~RNN.cpu
      ~RNN.cuda
      ~RNN.double
      ~RNN.eval
      ~RNN.extra_repr
      ~RNN.float
      ~RNN.forward
      ~RNN.freeze
      ~RNN.get_buffer
      ~RNN.get_extra_state
      ~RNN.get_from_queue
      ~RNN.get_parameter
      ~RNN.get_progress_bar_dict
      ~RNN.get_submodule
      ~RNN.half
      ~RNN.load_from_checkpoint
      ~RNN.load_state_dict
      ~RNN.log
      ~RNN.log_dict
      ~RNN.log_grad_norm
      ~RNN.lr_schedulers
      ~RNN.manual_backward
      ~RNN.modules
      ~RNN.named_buffers
      ~RNN.named_children
      ~RNN.named_modules
      ~RNN.named_parameters
      ~RNN.on_after_backward
      ~RNN.on_after_batch_transfer
      ~RNN.on_before_backward
      ~RNN.on_before_batch_transfer
      ~RNN.on_before_optimizer_step
      ~RNN.on_before_zero_grad
      ~RNN.on_epoch_end
      ~RNN.on_epoch_start
      ~RNN.on_fit_end
      ~RNN.on_fit_start
      ~RNN.on_hpc_load
      ~RNN.on_hpc_save
      ~RNN.on_load_checkpoint
      ~RNN.on_post_move_to_device
      ~RNN.on_predict_batch_end
      ~RNN.on_predict_batch_start
      ~RNN.on_predict_dataloader
      ~RNN.on_predict_end
      ~RNN.on_predict_epoch_end
      ~RNN.on_predict_epoch_start
      ~RNN.on_predict_model_eval
      ~RNN.on_predict_start
      ~RNN.on_pretrain_routine_end
      ~RNN.on_pretrain_routine_start
      ~RNN.on_save_checkpoint
      ~RNN.on_test_batch_end
      ~RNN.on_test_batch_start
      ~RNN.on_test_dataloader
      ~RNN.on_test_end
      ~RNN.on_test_epoch_end
      ~RNN.on_test_epoch_start
      ~RNN.on_test_model_eval
      ~RNN.on_test_model_train
      ~RNN.on_test_start
      ~RNN.on_train_batch_end
      ~RNN.on_train_batch_start
      ~RNN.on_train_dataloader
      ~RNN.on_train_end
      ~RNN.on_train_epoch_end
      ~RNN.on_train_epoch_start
      ~RNN.on_train_start
      ~RNN.on_val_dataloader
      ~RNN.on_validation_batch_end
      ~RNN.on_validation_batch_start
      ~RNN.on_validation_end
      ~RNN.on_validation_epoch_end
      ~RNN.on_validation_epoch_start
      ~RNN.on_validation_model_eval
      ~RNN.on_validation_model_train
      ~RNN.on_validation_start
      ~RNN.optimizer_step
      ~RNN.optimizer_zero_grad
      ~RNN.optimizers
      ~RNN.parameters
      ~RNN.predict_dataloader
      ~RNN.predict_step
      ~RNN.prepare_data
      ~RNN.print
      ~RNN.register_backward_hook
      ~RNN.register_buffer
      ~RNN.register_forward_hook
      ~RNN.register_forward_pre_hook
      ~RNN.register_full_backward_hook
      ~RNN.register_module
      ~RNN.register_parameter
      ~RNN.requires_grad_
      ~RNN.save_hyperparameters
      ~RNN.set_extra_state
      ~RNN.setup
      ~RNN.share_memory
      ~RNN.state_dict
      ~RNN.summarize
      ~RNN.summary
      ~RNN.tbptt_split_batch
      ~RNN.teardown
      ~RNN.test_dataloader
      ~RNN.test_epoch_end
      ~RNN.test_step
      ~RNN.test_step_end
      ~RNN.to
      ~RNN.to_empty
      ~RNN.to_onnx
      ~RNN.to_torchscript
      ~RNN.toggle_optimizer
      ~RNN.train
      ~RNN.train_dataloader
      ~RNN.training_epoch_end
      ~RNN.training_step
      ~RNN.training_step_end
      ~RNN.transfer_batch_to_device
      ~RNN.type
      ~RNN.unfreeze
      ~RNN.untoggle_optimizer
      ~RNN.val_dataloader
      ~RNN.validation_epoch_end
      ~RNN.validation_step
      ~RNN.validation_step_end
      ~RNN.xpu
      ~RNN.zero_grad





   .. rubric:: Attributes

   .. autosummary::

      ~RNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~RNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~RNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~RNN.T_destination
      ~RNN.automatic_optimization
      ~RNN.current_epoch
      ~RNN.device
      ~RNN.dtype
      ~RNN.dump_patches
      ~RNN.example_input_array
      ~RNN.global_rank
      ~RNN.global_step
      ~RNN.hparams
      ~RNN.hparams_initial
      ~RNN.loaded_optimizer_states_dict
      ~RNN.local_rank
      ~RNN.logger
      ~RNN.model_size
      ~RNN.on_gpu
      ~RNN.truncated_bptt_steps
      ~RNN.training
