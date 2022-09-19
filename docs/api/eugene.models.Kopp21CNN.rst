eugene.models.Kopp21CNN
=======================

.. currentmodule:: eugene.models

.. autoclass:: Kopp21CNN

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Kopp21CNN.__init__
      ~Kopp21CNN.add_module
      ~Kopp21CNN.add_to_queue
      ~Kopp21CNN.all_gather
      ~Kopp21CNN.apply
      ~Kopp21CNN.backward
      ~Kopp21CNN.bfloat16
      ~Kopp21CNN.buffers
      ~Kopp21CNN.children
      ~Kopp21CNN.clip_gradients
      ~Kopp21CNN.configure_callbacks
      ~Kopp21CNN.configure_gradient_clipping
      ~Kopp21CNN.configure_optimizers
      ~Kopp21CNN.configure_sharded_model
      ~Kopp21CNN.cpu
      ~Kopp21CNN.cuda
      ~Kopp21CNN.double
      ~Kopp21CNN.eval
      ~Kopp21CNN.extra_repr
      ~Kopp21CNN.float
      ~Kopp21CNN.forward
      ~Kopp21CNN.freeze
      ~Kopp21CNN.get_buffer
      ~Kopp21CNN.get_extra_state
      ~Kopp21CNN.get_from_queue
      ~Kopp21CNN.get_parameter
      ~Kopp21CNN.get_progress_bar_dict
      ~Kopp21CNN.get_submodule
      ~Kopp21CNN.half
      ~Kopp21CNN.load_from_checkpoint
      ~Kopp21CNN.load_state_dict
      ~Kopp21CNN.log
      ~Kopp21CNN.log_dict
      ~Kopp21CNN.log_grad_norm
      ~Kopp21CNN.lr_schedulers
      ~Kopp21CNN.manual_backward
      ~Kopp21CNN.modules
      ~Kopp21CNN.named_buffers
      ~Kopp21CNN.named_children
      ~Kopp21CNN.named_modules
      ~Kopp21CNN.named_parameters
      ~Kopp21CNN.on_after_backward
      ~Kopp21CNN.on_after_batch_transfer
      ~Kopp21CNN.on_before_backward
      ~Kopp21CNN.on_before_batch_transfer
      ~Kopp21CNN.on_before_optimizer_step
      ~Kopp21CNN.on_before_zero_grad
      ~Kopp21CNN.on_epoch_end
      ~Kopp21CNN.on_epoch_start
      ~Kopp21CNN.on_fit_end
      ~Kopp21CNN.on_fit_start
      ~Kopp21CNN.on_hpc_load
      ~Kopp21CNN.on_hpc_save
      ~Kopp21CNN.on_load_checkpoint
      ~Kopp21CNN.on_post_move_to_device
      ~Kopp21CNN.on_predict_batch_end
      ~Kopp21CNN.on_predict_batch_start
      ~Kopp21CNN.on_predict_dataloader
      ~Kopp21CNN.on_predict_end
      ~Kopp21CNN.on_predict_epoch_end
      ~Kopp21CNN.on_predict_epoch_start
      ~Kopp21CNN.on_predict_model_eval
      ~Kopp21CNN.on_predict_start
      ~Kopp21CNN.on_pretrain_routine_end
      ~Kopp21CNN.on_pretrain_routine_start
      ~Kopp21CNN.on_save_checkpoint
      ~Kopp21CNN.on_test_batch_end
      ~Kopp21CNN.on_test_batch_start
      ~Kopp21CNN.on_test_dataloader
      ~Kopp21CNN.on_test_end
      ~Kopp21CNN.on_test_epoch_end
      ~Kopp21CNN.on_test_epoch_start
      ~Kopp21CNN.on_test_model_eval
      ~Kopp21CNN.on_test_model_train
      ~Kopp21CNN.on_test_start
      ~Kopp21CNN.on_train_batch_end
      ~Kopp21CNN.on_train_batch_start
      ~Kopp21CNN.on_train_dataloader
      ~Kopp21CNN.on_train_end
      ~Kopp21CNN.on_train_epoch_end
      ~Kopp21CNN.on_train_epoch_start
      ~Kopp21CNN.on_train_start
      ~Kopp21CNN.on_val_dataloader
      ~Kopp21CNN.on_validation_batch_end
      ~Kopp21CNN.on_validation_batch_start
      ~Kopp21CNN.on_validation_end
      ~Kopp21CNN.on_validation_epoch_end
      ~Kopp21CNN.on_validation_epoch_start
      ~Kopp21CNN.on_validation_model_eval
      ~Kopp21CNN.on_validation_model_train
      ~Kopp21CNN.on_validation_start
      ~Kopp21CNN.optimizer_step
      ~Kopp21CNN.optimizer_zero_grad
      ~Kopp21CNN.optimizers
      ~Kopp21CNN.parameters
      ~Kopp21CNN.predict_dataloader
      ~Kopp21CNN.predict_step
      ~Kopp21CNN.prepare_data
      ~Kopp21CNN.print
      ~Kopp21CNN.register_backward_hook
      ~Kopp21CNN.register_buffer
      ~Kopp21CNN.register_forward_hook
      ~Kopp21CNN.register_forward_pre_hook
      ~Kopp21CNN.register_full_backward_hook
      ~Kopp21CNN.register_module
      ~Kopp21CNN.register_parameter
      ~Kopp21CNN.requires_grad_
      ~Kopp21CNN.save_hyperparameters
      ~Kopp21CNN.set_extra_state
      ~Kopp21CNN.setup
      ~Kopp21CNN.share_memory
      ~Kopp21CNN.state_dict
      ~Kopp21CNN.summarize
      ~Kopp21CNN.summary
      ~Kopp21CNN.tbptt_split_batch
      ~Kopp21CNN.teardown
      ~Kopp21CNN.test_dataloader
      ~Kopp21CNN.test_epoch_end
      ~Kopp21CNN.test_step
      ~Kopp21CNN.test_step_end
      ~Kopp21CNN.to
      ~Kopp21CNN.to_empty
      ~Kopp21CNN.to_onnx
      ~Kopp21CNN.to_torchscript
      ~Kopp21CNN.toggle_optimizer
      ~Kopp21CNN.train
      ~Kopp21CNN.train_dataloader
      ~Kopp21CNN.training_epoch_end
      ~Kopp21CNN.training_step
      ~Kopp21CNN.training_step_end
      ~Kopp21CNN.transfer_batch_to_device
      ~Kopp21CNN.type
      ~Kopp21CNN.unfreeze
      ~Kopp21CNN.untoggle_optimizer
      ~Kopp21CNN.val_dataloader
      ~Kopp21CNN.validation_epoch_end
      ~Kopp21CNN.validation_step
      ~Kopp21CNN.validation_step_end
      ~Kopp21CNN.xpu
      ~Kopp21CNN.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Kopp21CNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~Kopp21CNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~Kopp21CNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~Kopp21CNN.T_destination
      ~Kopp21CNN.automatic_optimization
      ~Kopp21CNN.current_epoch
      ~Kopp21CNN.device
      ~Kopp21CNN.dtype
      ~Kopp21CNN.dump_patches
      ~Kopp21CNN.example_input_array
      ~Kopp21CNN.global_rank
      ~Kopp21CNN.global_step
      ~Kopp21CNN.hparams
      ~Kopp21CNN.hparams_initial
      ~Kopp21CNN.loaded_optimizer_states_dict
      ~Kopp21CNN.local_rank
      ~Kopp21CNN.logger
      ~Kopp21CNN.model_size
      ~Kopp21CNN.on_gpu
      ~Kopp21CNN.truncated_bptt_steps
      ~Kopp21CNN.training
   
   