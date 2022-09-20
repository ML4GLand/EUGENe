:github_url: eugene.models.RNN

eugene.models.RNN
=================

.. currentmodule:: eugene.models

.. autoclass:: RNN



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~eugene.models.RNN.CHECKPOINT_HYPER_PARAMS_KEY
      ~eugene.models.RNN.CHECKPOINT_HYPER_PARAMS_NAME
      ~eugene.models.RNN.CHECKPOINT_HYPER_PARAMS_TYPE
      ~eugene.models.RNN.T_destination
      ~eugene.models.RNN.automatic_optimization
      ~eugene.models.RNN.current_epoch
      ~eugene.models.RNN.device
      ~eugene.models.RNN.dtype
      ~eugene.models.RNN.dump_patches
      ~eugene.models.RNN.example_input_array
      ~eugene.models.RNN.global_rank
      ~eugene.models.RNN.global_step
      ~eugene.models.RNN.hparams
      ~eugene.models.RNN.hparams_initial
      ~eugene.models.RNN.loaded_optimizer_states_dict
      ~eugene.models.RNN.local_rank
      ~eugene.models.RNN.logger
      ~eugene.models.RNN.model_size
      ~eugene.models.RNN.on_gpu
      ~eugene.models.RNN.truncated_bptt_steps
      ~eugene.models.RNN.training





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~eugene.models.RNN.add_module
      ~eugene.models.RNN.add_to_queue
      ~eugene.models.RNN.all_gather
      ~eugene.models.RNN.apply
      ~eugene.models.RNN.backward
      ~eugene.models.RNN.bfloat16
      ~eugene.models.RNN.buffers
      ~eugene.models.RNN.children
      ~eugene.models.RNN.clip_gradients
      ~eugene.models.RNN.configure_callbacks
      ~eugene.models.RNN.configure_gradient_clipping
      ~eugene.models.RNN.configure_optimizers
      ~eugene.models.RNN.configure_sharded_model
      ~eugene.models.RNN.cpu
      ~eugene.models.RNN.cuda
      ~eugene.models.RNN.double
      ~eugene.models.RNN.eval
      ~eugene.models.RNN.extra_repr
      ~eugene.models.RNN.float
      ~eugene.models.RNN.forward
      ~eugene.models.RNN.freeze
      ~eugene.models.RNN.get_buffer
      ~eugene.models.RNN.get_extra_state
      ~eugene.models.RNN.get_from_queue
      ~eugene.models.RNN.get_parameter
      ~eugene.models.RNN.get_progress_bar_dict
      ~eugene.models.RNN.get_submodule
      ~eugene.models.RNN.half
      ~eugene.models.RNN.load_from_checkpoint
      ~eugene.models.RNN.load_state_dict
      ~eugene.models.RNN.log
      ~eugene.models.RNN.log_dict
      ~eugene.models.RNN.log_grad_norm
      ~eugene.models.RNN.lr_schedulers
      ~eugene.models.RNN.manual_backward
      ~eugene.models.RNN.modules
      ~eugene.models.RNN.named_buffers
      ~eugene.models.RNN.named_children
      ~eugene.models.RNN.named_modules
      ~eugene.models.RNN.named_parameters
      ~eugene.models.RNN.on_after_backward
      ~eugene.models.RNN.on_after_batch_transfer
      ~eugene.models.RNN.on_before_backward
      ~eugene.models.RNN.on_before_batch_transfer
      ~eugene.models.RNN.on_before_optimizer_step
      ~eugene.models.RNN.on_before_zero_grad
      ~eugene.models.RNN.on_epoch_end
      ~eugene.models.RNN.on_epoch_start
      ~eugene.models.RNN.on_fit_end
      ~eugene.models.RNN.on_fit_start
      ~eugene.models.RNN.on_hpc_load
      ~eugene.models.RNN.on_hpc_save
      ~eugene.models.RNN.on_load_checkpoint
      ~eugene.models.RNN.on_post_move_to_device
      ~eugene.models.RNN.on_predict_batch_end
      ~eugene.models.RNN.on_predict_batch_start
      ~eugene.models.RNN.on_predict_dataloader
      ~eugene.models.RNN.on_predict_end
      ~eugene.models.RNN.on_predict_epoch_end
      ~eugene.models.RNN.on_predict_epoch_start
      ~eugene.models.RNN.on_predict_model_eval
      ~eugene.models.RNN.on_predict_start
      ~eugene.models.RNN.on_pretrain_routine_end
      ~eugene.models.RNN.on_pretrain_routine_start
      ~eugene.models.RNN.on_save_checkpoint
      ~eugene.models.RNN.on_test_batch_end
      ~eugene.models.RNN.on_test_batch_start
      ~eugene.models.RNN.on_test_dataloader
      ~eugene.models.RNN.on_test_end
      ~eugene.models.RNN.on_test_epoch_end
      ~eugene.models.RNN.on_test_epoch_start
      ~eugene.models.RNN.on_test_model_eval
      ~eugene.models.RNN.on_test_model_train
      ~eugene.models.RNN.on_test_start
      ~eugene.models.RNN.on_train_batch_end
      ~eugene.models.RNN.on_train_batch_start
      ~eugene.models.RNN.on_train_dataloader
      ~eugene.models.RNN.on_train_end
      ~eugene.models.RNN.on_train_epoch_end
      ~eugene.models.RNN.on_train_epoch_start
      ~eugene.models.RNN.on_train_start
      ~eugene.models.RNN.on_val_dataloader
      ~eugene.models.RNN.on_validation_batch_end
      ~eugene.models.RNN.on_validation_batch_start
      ~eugene.models.RNN.on_validation_end
      ~eugene.models.RNN.on_validation_epoch_end
      ~eugene.models.RNN.on_validation_epoch_start
      ~eugene.models.RNN.on_validation_model_eval
      ~eugene.models.RNN.on_validation_model_train
      ~eugene.models.RNN.on_validation_start
      ~eugene.models.RNN.optimizer_step
      ~eugene.models.RNN.optimizer_zero_grad
      ~eugene.models.RNN.optimizers
      ~eugene.models.RNN.parameters
      ~eugene.models.RNN.predict_dataloader
      ~eugene.models.RNN.predict_step
      ~eugene.models.RNN.prepare_data
      ~eugene.models.RNN.print
      ~eugene.models.RNN.register_backward_hook
      ~eugene.models.RNN.register_buffer
      ~eugene.models.RNN.register_forward_hook
      ~eugene.models.RNN.register_forward_pre_hook
      ~eugene.models.RNN.register_full_backward_hook
      ~eugene.models.RNN.register_module
      ~eugene.models.RNN.register_parameter
      ~eugene.models.RNN.requires_grad_
      ~eugene.models.RNN.save_hyperparameters
      ~eugene.models.RNN.set_extra_state
      ~eugene.models.RNN.setup
      ~eugene.models.RNN.share_memory
      ~eugene.models.RNN.state_dict
      ~eugene.models.RNN.summarize
      ~eugene.models.RNN.summary
      ~eugene.models.RNN.tbptt_split_batch
      ~eugene.models.RNN.teardown
      ~eugene.models.RNN.test_dataloader
      ~eugene.models.RNN.test_epoch_end
      ~eugene.models.RNN.test_step
      ~eugene.models.RNN.test_step_end
      ~eugene.models.RNN.to
      ~eugene.models.RNN.to_empty
      ~eugene.models.RNN.to_onnx
      ~eugene.models.RNN.to_torchscript
      ~eugene.models.RNN.toggle_optimizer
      ~eugene.models.RNN.train
      ~eugene.models.RNN.train_dataloader
      ~eugene.models.RNN.training_epoch_end
      ~eugene.models.RNN.training_step
      ~eugene.models.RNN.training_step_end
      ~eugene.models.RNN.transfer_batch_to_device
      ~eugene.models.RNN.type
      ~eugene.models.RNN.unfreeze
      ~eugene.models.RNN.untoggle_optimizer
      ~eugene.models.RNN.val_dataloader
      ~eugene.models.RNN.validation_epoch_end
      ~eugene.models.RNN.validation_step
      ~eugene.models.RNN.validation_step_end
      ~eugene.models.RNN.xpu
      ~eugene.models.RNN.zero_grad



.. _sphx_glr_backref_eugene.models.RNN:
