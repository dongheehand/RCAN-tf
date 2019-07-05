import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
import tqdm
import skimage.measure
import skimage.color
import pickle


def train(args, model, sess):
    
    '''
    If you want to fine-tuning from pre-trained model,
    You should --fine_tuning option to True and --pre_trained_model option to the pre-trained model path
    '''
    
    if args.fine_tuning :
        
        if args.load_tail_part:
            variables_to_restore = [var for var in tf.global_variables()]
        else:
            variables_to_restore = [var for var in tf.global_variables() if 'up_sample' not in var.name and 'conv_rec' not in var.name]
            
        loader = tf.train.Saver(variables_to_restore)
        loader.restore(sess, args.pre_trained_model)
        
        print("saved model is loaded for fine-tuning!")
        if not args.load_tail_part:
            print("Tail part is not loaded!")
        print("model path is %s"%(args.pre_trained_model))
        
    num_imgs = len(os.listdir(args.train_GT_path))
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs',sess.graph)
    if args.test_with_train:
        f = open("RCAN_X%d_train_log.txt"%(args.scale), 'w')
        model_config = 'scale : %d \t n_feats : %d \t n_RG : %d \t n_RCAB : %d \n'%(args.scale, args.n_feats, args.n_RG, args.n_RCAB)
        f.write(model_config)
        val_num = len(os.listdir(args.test_LR_path))
    
    count = 0
    step = num_imgs // args.batch_size
    saver = tf.train.Saver(max_to_keep = None)
    
    '''
    If your train data is small enough for fitting in main memory,
    It is better to set --in_memory option to True
    
    '''
    if args.in_memory:
        
        lr_imgs = util.image_loader(args.train_LR_path)
        gt_imgs = util.image_loader(args.train_GT_path)
        
        if args.test_with_train:
            val_lr_imgs = util.image_loader(args.test_LR_path)
            val_gt_imgs = util.image_loader(args.test_GT_path)
        
        while count < args.max_step:
            random_index = np.random.permutation(len(lr_imgs))
            for k in range(step):
                s_time = time.time()
                lr_batch, gt_batch = util.batch_gen(lr_imgs, gt_imgs, args.patch_size, args.scale, args.batch_size, random_index, k)
                _, losses = sess.run([model.train, model.loss], feed_dict = {model.LR : lr_batch, model.GT : gt_batch, model.global_step : count})
                count += 1
                e_time = time.time()
                if count % args.log_freq == 0:
                    summary = sess.run(merged, feed_dict = {model.LR : lr_batch, model.GT: gt_batch})
                    train_writer.add_summary(summary, count)
                    
                    if args.test_with_train:
                        util.train_with_test(args, model, sess, saver, f, count, val_lr_imgs, val_gt_imgs)
                        f.close()
                        f = open("RCAN_X%d_train_log.txt"%(args.scale), 'a')
                    
                    print("%d training step completed" % count)
                    print("Loss : %0.4f"%losses)
                    print("Elpased time : %0.4f"%(e_time - s_time))
                    
                if ((count) % args.model_save_freq ==0):
                    saver.save(sess, os.path.join(args.model_path,'RCAN_X%d_%d_%d_%d'%(args.scale, args.n_feats, args.n_RG, args.n_RCAB)),global_step = count, write_meta_graph = False)
    
        saver.save(sess, os.path.join(args.model_path,'RCAN_X%d_%d_%d_%d_last'%(args.scale, args.n_feats, args.n_RG, args.n_RCAB)),global_step = count, write_meta_graph = False)
    
    
    else:
        
        while count < args.max_step:
            sess.run(model.data_loader.init_op['tr_init'])
            
            for k in range(step):
                _ = sess.run([model.train], feed_dict = {model.global_step : count})
                count += 1
                if count % args.log_freq == 0:
                    summary, loss = sess.run([merged, model.loss])
                    train_writer.add_summary(summary, count)
                    
                    if args.test_with_train:
                        util.train_with_test(args, model, sess, saver, f, count, None, None, val_num = val_num)
                        sess.run(model.data_loader.init_op['tr_init'])

                    print("%d training step completed" % count)
                    print("Loss : %0.4f"%losses)
                    print("Elpased time : %0.4f"%(e_time - s_time))

                if ((count) % args.model_save_freq ==0):
                    saver.save(sess, os.path.join(args.model_path,'RCA_model_%04d_feats_%02d_res_%0.2f_scale'%(args.n_feats,args.n_RG,args.scale)),global_step = count, write_meta_graph = False)

    saver.save(sess, os.path.join(args.model_path,'RCA_model_%04d_feats_%02d_res_%0.2f_scale_last'%(args.n_feats,args.n_RG,args.scale)))
    
    if args.test_with_train:
        f.close()

def test(args, model, sess):
    
    loader = tf.train.Saver(max_to_keep = None)
    
    loader.restore(sess, args.pre_trained_model)
    print("saved model is loaded for test!")
    print("model path is %s"%args.pre_trained_model)
    
    val_LR = sorted(os.listdir(args.test_LR_path))
    val_HR = sorted(os.listdir(args.test_GT_path))
    
    val_LR_imgs = util.image_loader(args.test_LR_path)
    val_GT_imgs = util.image_loader(args.test_GT_path)
    
    Y_PSNR_list = []
    Y_SSIM_list = []
    
    file = open('./RCAN_X%d_%s_result.txt'%(args.scale, args.test_set), 'w')
    
    if args.in_memory:

        for i, img_LR in enumerate(val_LR_imgs):
            
            batch_img_LR = np.expand_dims(img_LR, axis = 0)
            img_HR = val_GT_imgs[i]
            
            if args.self_ensemble:
                output = util.self_ensemble(args, model, sess, batch_img_LR, is_recursive = args.chop_forward)
            
            else:
                if args.chop_forward:
                    output = util.recursive_forwarding(batch_img_LR, args.scale, args.chop_size, sess, model, args.chop_shave)
                    output = output[0]
                else:
                    output = sess.run(model.output, feed_dict = {model.LR : batch_img_LR})
                    output = output[0]
                    
            
            h, w, c = output.shape
            val_gt = img_HR[:h,:w]
           
            y_psnr, y_ssim = util.compare_measure(val_gt, output, args)
            
            Y_PSNR_list.append(y_psnr)
            Y_SSIM_list.append(y_ssim)
            file.write('file name : %s PSNR : %04f SSIM : %04f \n'%(val_LR[i], y_psnr, y_ssim))
            
            if args.save_test_result :
                im = Image.fromarray(output)
                split_name = val_LR[i].split(".")
                im.save(os.path.join(args.result_path,"%sX%d.%s"%(''.join(map(str, split_name[:-1])), args.scale, split_name[-1])))

    
    else:
        
        sess.run(model.data_loader.init_op['val_init'])
        
        for i in range(len(val_LR)):

            output, val_gt = sess.run([model.output, model.label])
            output = output[0]
            val_gt = val_gt[0]
            h, w, c = output.shape
            val_gt = val_gt[:h,:w]
            
            y_psnr, y_ssim = util.compare_measure(val_gt, output, args)
            
            Y_PSNR_list.append(y_psnr)
            Y_SSIM_list.append(y_ssim)
            file.write('file name : %s PSNR : %04f SSIM : %04f'%(val_LR[i], y_psnr, y_ssim))

            if args.save_test_result:
                im = Image.fromarray(output)
                split_name = val_LR[i].split(".")
                im.save(os.path.join(args.result_path,"%sX%d.%s"%(''.join(map(str, split_name[:-1])), args.scale, split_name[-1])))
    
    length = len(val_LR)
    mean_Y_PSNR = sum(Y_PSNR_list) / length
    mean_SSIM = sum(Y_SSIM_list) / length
    
    file.write("Y_PSNR : %0.4f SSIM : %0.4f \n"%(mean_Y_PSNR, mean_SSIM))
    file.close()
    
def test_only(args, model, sess):
    
    loader = tf.train.Saver(max_to_keep = None)
    loader.restore(sess, args.pre_trained_model)
    
    print("saved model is loaded for test only!")
    print("model path is %s"%args.pre_trained_model)
    
    val_LR = sorted(os.listdir(args.test_LR_path))
    val_LR_imgs = util.image_loader(args.test_LR_path)
    
    if args.in_memory:

        for i, img_LR in enumerate(val_LR_imgs):
            
            batch_img_LR = np.expand_dims(img_LR, axis = 0)
            
            if args.self_ensemble:
                output = util.self_ensemble(args, model, sess, batch_img_LR, is_recursive = args.chop_forward)
            
            else:
                if args.chop_forward:
                    output = util.recursive_forwarding(batch_img_LR, args.scale, args.chop_size, sess, model, args.chop_shave)
                    output = output[0]
                else:
                    output = sess.run(model.output, feed_dict = {model.LR : batch_img_LR})
                    output = output[0]
                    
            im = Image.fromarray(output)
            split_name = val_LR[i].split(".")
            im.save(os.path.join(args.result_path,"%sX%d.%s"%(''.join(map(str, split_name[:-1])), args.scale, split_name[-1])))

    
    else:
        
        sess.run(model.data_loader.init_op['val_init'])
        
        for i in range(len(val_LR)):
            output = sess.run([model.output])
            output = output[0]
            
            im = Image.fromarray(output)
            split_name = val_LR[i].split(".")
            im.save(os.path.join(args.result_path,"%sX%d.%s"%(''.join(map(str, split_name[:-1])), args.scale, split_name[-1])))

