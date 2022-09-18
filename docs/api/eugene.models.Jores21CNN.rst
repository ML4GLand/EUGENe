eugene.models.Jores21CNN
========================

.. currentmodule:: eugene.models

.. autoclass:: Jores21CNN


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~Jores21CNN.__init__
      ~Jores21CNN.add_module
      ~Jores21CNN.add_to_queue
      ~Jores21CNN.all_gather
      ~Jores21CNN.apply
      ~Jores21CNN.backward
      ~Jores21CNN.bfloat16
      ~Jores21CNN.buffers
      ~Jores21CNN.children
      ~Jores21CNN.clip_gradients
      ~Jores21CNN.configure_callbacks
      ~Jores21CNN.configure_gradient_clipping
      ~Jores21CNN.configure_optimizers
      ~Jores21CNN.configure_sharded_model
      ~Jores21CNN.cpu
      ~Jores21CNN.cuda
      ~Jores21CNN.double
      ~Jores21CNN.eval
      ~Jores21CNN.extra_repr
      ~Jores21CNN.float
      ~Jores21CNN.forward
      ~Jores21CNN.freeze
      ~Jores21CNN.get_buffer
      ~Jores21CNN.get_extra_state
      ~Jores21CNN.get_from_queue
      ~Jores21CNN.get_parameter
      ~Jores21CNN.get_progress_bar_dict
      ~Jores21CNN.get_submodule
      ~Jores21CNN.half
      ~Jores21CNN.load_from_checkpoint
      ~Jores21CNN.load_state_dict
      ~Jores21CNN.log
      ~Jores21CNN.log_dict
      ~Jores21CNN.log_grad_norm
      ~Jores21CNN.lr_schedulers
      ~Jores21CNN.manual_backward
      ~Jores21CNN.modules
      ~Jores21CNN.named_buffers
      ~Jores21CNN.named_children
      ~Jores21CNN.named_modules
      ~Jores21CNN.named_parameters
      ~Jores21CNN.on_after_backward
      ~Jores21CNN.on_after_batch_transfer
      ~Jores21CNN.on_before_backward
      ~Jores21CNN.on_before_batch_transfer
      ~Jores21CNN.on_before_optimizer_step
      ~Jores21CNN.on_before_zero_grad
      ~Jores21CNN.on_epoch_end
      ~Jores21CNN.on_epoch_start
      ~Jores21CNN.on_fit_end
      ~Jores21CNN.on_fit_start
      ~Jores21CNN.on_hpc_load
      ~Jores21CNN.on_hpc_save
      ~Jores21CNN.on_load_checkpoint
      ~Jores21CNN.on_post_move_to_device
      ~Jores21CNN.on_predict_batch_end
      ~Jores21CNN.on_predict_batch_start
      ~Jores21CNN.on_predict_dataloader
      ~Jores21CNN.on_predict_end
      ~Jores21CNN.on_predict_epoch_end
      ~Jores21CNN.on_predict_epoch_start
      ~Jores21CNN.on_predict_model_eval
      ~Jores21CNN.on_predict_start
      ~Jores21CNN.on_pretrain_routine_end
      ~Jores21CNN.on_pretrain_routine_start
      ~Jores21CNN.on_save_checkpoint
      ~Jores21CNN.on_test_batch_end
      ~Jores21CNN.on_test_batch_start
      ~Jores21CNN.on_test_dataloader
      ~Jores21CNN.on_test_end
      ~Jores21CNN.on_test_epoch_end
      ~Jores21CNN.on_test_epoch_start
      ~Jores21CNN.on_test_model_eval
      ~Jores21CNN.on_test_model_train
      ~Jores21CNN.on_test_start
      ~Jores21CNN.on_train_batch_end
      ~Jores21CNN.on_train_batch_start
      ~Jores21CNN.on_train_dataloader
      ~Jores21CNN.on_train_end
      ~Jores21CNN.on_train_epoch_end
      ~Jores21CNN.on_train_epoch_start
      ~Jores21CNN.on_train_start
      ~Jores21CNN.on_val_dataloader
      ~Jores21CNN.on_validation_batch_end
      ~Jores21CNN.on_validation_batch_start
      ~Jores21CNN.on_validation_end
      ~Jores21CNN.on_validation_epoch_end
      ~Jores21CNN.on_validation_epoch_start
      ~Jores21CNN.on_validation_model_eval
      ~Jores21CNN.on_validation_model_train
      ~Jores21CNN.on_validation_start
      ~Jores21CNN.optimizer_step
      ~Jores21CNN.optimizer_zero_grad
      ~Jores21CNN.optimizers
      ~Jores21CNN.parameters
      ~Jores21CNN.predict_dataloader
      ~Jores21CNN.predict_step
      ~Jores21CNN.prepare_data
      ~Jores21CNN.print
      ~Jores21CNN.register_backward_hook
      ~Jores21CNN.register_buffer
      ~Jores21CNN.register_forward_hook
      ~Jores21CNN.register_forward_pre_hook
      ~Jores21CNN.register_full_backward_hook
      ~Jores21CNN.register_module
      ~Jores21CNN.register_parameter
      ~Jores21CNN.requires_grad_
      ~Jores21CNN.save_hyperparameters
      ~Jores21CNN.set_extra_state
      ~Jores21CNN.setup
      ~Jores21CNN.share_memory
      ~Jores21CNN.state_dict
      ~Jores21CNN.summarize
      ~Jores21CNN.summary
      ~Jores21CNN.tbptt_split_batch
      ~Jores21CNN.teardown
      ~Jores21CNN.test_dataloader
      ~Jores21CNN.test_epoch_end
      ~Jores21CNN.test_step
      ~Jores21CNN.test_step_end
      ~Jores21CNN.to
      ~Jores21CNN.to_empty
      ~Jores21CNN.to_onnx
      ~Jores21CNN.to_torchscript
      ~Jores21CNN.toggle_optimizer
      ~Jores21CNN.train
      ~Jores21CNN.train_dataloader
      ~Jores21CNN.training_epoch_end
      ~Jores21CNN.training_step
      ~Jores21CNN.training_step_end
      ~Jores21CNN.transfer_batch_to_device
      ~Jores21CNN.type
      ~Jores21CNN.unfreeze
      ~Jores21CNN.untoggle_optimizer
      ~Jores21CNN.val_dataloader
      ~Jores21CNN.validation_epoch_end
      ~Jores21CNN.validation_step
      ~Jores21CNN.validation_step_end
      ~Jores21CNN.xpu
      ~Jores21CNN.zero_grad





   .. rubric:: Attributes

   .. autosummary::

      ~Jores21CNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~Jores21CNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~Jores21CNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~Jores21CNN.T_destination
      ~Jores21CNN.automatic_optimization
      ~Jores21CNN.current_epoch
      ~Jores21CNN.device
      ~Jores21CNN.dtype
      ~Jores21CNN.dump_patches
      ~Jores21CNN.example_input_array
      ~Jores21CNN.global_rank
      ~Jores21CNN.global_step
      ~Jores21CNN.hparams
      ~Jores21CNN.hparams_initial
      ~Jores21CNN.loaded_optimizer_states_dict
      ~Jores21CNN.local_rank
      ~Jores21CNN.logger
      ~Jores21CNN.model_size
      ~Jores21CNN.on_gpu
      ~Jores21CNN.truncated_bptt_steps
      ~Jores21CNN.training
