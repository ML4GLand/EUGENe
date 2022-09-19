eugene.models.Hybrid
====================

.. currentmodule:: eugene.models

.. autoclass:: Hybrid

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Hybrid.__init__
      ~Hybrid.add_module
      ~Hybrid.add_to_queue
      ~Hybrid.all_gather
      ~Hybrid.apply
      ~Hybrid.backward
      ~Hybrid.bfloat16
      ~Hybrid.buffers
      ~Hybrid.children
      ~Hybrid.clip_gradients
      ~Hybrid.configure_callbacks
      ~Hybrid.configure_gradient_clipping
      ~Hybrid.configure_optimizers
      ~Hybrid.configure_sharded_model
      ~Hybrid.cpu
      ~Hybrid.cuda
      ~Hybrid.double
      ~Hybrid.eval
      ~Hybrid.extra_repr
      ~Hybrid.float
      ~Hybrid.forward
      ~Hybrid.freeze
      ~Hybrid.get_buffer
      ~Hybrid.get_extra_state
      ~Hybrid.get_from_queue
      ~Hybrid.get_parameter
      ~Hybrid.get_progress_bar_dict
      ~Hybrid.get_submodule
      ~Hybrid.half
      ~Hybrid.load_from_checkpoint
      ~Hybrid.load_state_dict
      ~Hybrid.log
      ~Hybrid.log_dict
      ~Hybrid.log_grad_norm
      ~Hybrid.lr_schedulers
      ~Hybrid.manual_backward
      ~Hybrid.modules
      ~Hybrid.named_buffers
      ~Hybrid.named_children
      ~Hybrid.named_modules
      ~Hybrid.named_parameters
      ~Hybrid.on_after_backward
      ~Hybrid.on_after_batch_transfer
      ~Hybrid.on_before_backward
      ~Hybrid.on_before_batch_transfer
      ~Hybrid.on_before_optimizer_step
      ~Hybrid.on_before_zero_grad
      ~Hybrid.on_epoch_end
      ~Hybrid.on_epoch_start
      ~Hybrid.on_fit_end
      ~Hybrid.on_fit_start
      ~Hybrid.on_hpc_load
      ~Hybrid.on_hpc_save
      ~Hybrid.on_load_checkpoint
      ~Hybrid.on_post_move_to_device
      ~Hybrid.on_predict_batch_end
      ~Hybrid.on_predict_batch_start
      ~Hybrid.on_predict_dataloader
      ~Hybrid.on_predict_end
      ~Hybrid.on_predict_epoch_end
      ~Hybrid.on_predict_epoch_start
      ~Hybrid.on_predict_model_eval
      ~Hybrid.on_predict_start
      ~Hybrid.on_pretrain_routine_end
      ~Hybrid.on_pretrain_routine_start
      ~Hybrid.on_save_checkpoint
      ~Hybrid.on_test_batch_end
      ~Hybrid.on_test_batch_start
      ~Hybrid.on_test_dataloader
      ~Hybrid.on_test_end
      ~Hybrid.on_test_epoch_end
      ~Hybrid.on_test_epoch_start
      ~Hybrid.on_test_model_eval
      ~Hybrid.on_test_model_train
      ~Hybrid.on_test_start
      ~Hybrid.on_train_batch_end
      ~Hybrid.on_train_batch_start
      ~Hybrid.on_train_dataloader
      ~Hybrid.on_train_end
      ~Hybrid.on_train_epoch_end
      ~Hybrid.on_train_epoch_start
      ~Hybrid.on_train_start
      ~Hybrid.on_val_dataloader
      ~Hybrid.on_validation_batch_end
      ~Hybrid.on_validation_batch_start
      ~Hybrid.on_validation_end
      ~Hybrid.on_validation_epoch_end
      ~Hybrid.on_validation_epoch_start
      ~Hybrid.on_validation_model_eval
      ~Hybrid.on_validation_model_train
      ~Hybrid.on_validation_start
      ~Hybrid.optimizer_step
      ~Hybrid.optimizer_zero_grad
      ~Hybrid.optimizers
      ~Hybrid.parameters
      ~Hybrid.predict_dataloader
      ~Hybrid.predict_step
      ~Hybrid.prepare_data
      ~Hybrid.print
      ~Hybrid.register_backward_hook
      ~Hybrid.register_buffer
      ~Hybrid.register_forward_hook
      ~Hybrid.register_forward_pre_hook
      ~Hybrid.register_full_backward_hook
      ~Hybrid.register_module
      ~Hybrid.register_parameter
      ~Hybrid.requires_grad_
      ~Hybrid.save_hyperparameters
      ~Hybrid.set_extra_state
      ~Hybrid.setup
      ~Hybrid.share_memory
      ~Hybrid.state_dict
      ~Hybrid.summarize
      ~Hybrid.summary
      ~Hybrid.tbptt_split_batch
      ~Hybrid.teardown
      ~Hybrid.test_dataloader
      ~Hybrid.test_epoch_end
      ~Hybrid.test_step
      ~Hybrid.test_step_end
      ~Hybrid.to
      ~Hybrid.to_empty
      ~Hybrid.to_onnx
      ~Hybrid.to_torchscript
      ~Hybrid.toggle_optimizer
      ~Hybrid.train
      ~Hybrid.train_dataloader
      ~Hybrid.training_epoch_end
      ~Hybrid.training_step
      ~Hybrid.training_step_end
      ~Hybrid.transfer_batch_to_device
      ~Hybrid.type
      ~Hybrid.unfreeze
      ~Hybrid.untoggle_optimizer
      ~Hybrid.val_dataloader
      ~Hybrid.validation_epoch_end
      ~Hybrid.validation_step
      ~Hybrid.validation_step_end
      ~Hybrid.xpu
      ~Hybrid.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Hybrid.CHECKPOINT_HYPER_PARAMS_KEY
      ~Hybrid.CHECKPOINT_HYPER_PARAMS_NAME
      ~Hybrid.CHECKPOINT_HYPER_PARAMS_TYPE
      ~Hybrid.T_destination
      ~Hybrid.automatic_optimization
      ~Hybrid.current_epoch
      ~Hybrid.device
      ~Hybrid.dtype
      ~Hybrid.dump_patches
      ~Hybrid.example_input_array
      ~Hybrid.global_rank
      ~Hybrid.global_step
      ~Hybrid.hparams
      ~Hybrid.hparams_initial
      ~Hybrid.loaded_optimizer_states_dict
      ~Hybrid.local_rank
      ~Hybrid.logger
      ~Hybrid.model_size
      ~Hybrid.on_gpu
      ~Hybrid.truncated_bptt_steps
      ~Hybrid.training
   
   