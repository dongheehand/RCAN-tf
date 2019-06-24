import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
import skimage.measure
import skimage.color


def train(args, model, sess, saver):
    
    '''
    If you want to fine-tuning from pre-trained model,
    You should --fine_tuning option to True and --pre_trained_model option to the pre-trained model path
    '''
    
    if args.fine_tuning :
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for fine-tuning!")
        print("model path is %s"%(args.pre_trained_model))
        
    num_imgs = len(os.listdir(args.train_GT_path))
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs',sess.graph)
    if args.test_with_train:
        f = open("train_log_%02d_res_%02d_feats.txt"%(args.n_RG,args.n_feats), 'w')
        val_num = len(os.listdir(args.test_LR_path))
    
    count = 0
    step = num_imgs // args.batch_size
    
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
                        f = open("train_log_%02d_res_%02d_feats.txt"%(args.n_RG,args.n_feats), 'a')
                    print("%d training step completed" % count)
                    print("Loss : %0.4f"%losses)
                    print("Elpased time : %0.4f"%(e_time - s_time))
                if ((count) % args.model_save_freq ==0):
                    saver.save(sess, os.path.join(args.model_path,'RCA_model_%04d_feats_%02d_res_%0.2f_scale'%(args.n_feats,args.n_RG, args.scale)),global_step = count, write_meta_graph = False)
    
        saver.save(sess, os.path.join(args.model_path,'RCA_model_%04d_feats_%02d_res_%0.2f_scale'%(args.n_feats,args.n_RG,args.scale)),global_step = count, write_meta_graph = False)
    
    
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

def test(args, model, sess, saver, file, step = -1, loading = False):
        
    if loading:
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for test!")
        print("model path is %s"%args.pre_trained_model)
    
    val_LR = sorted(os.listdir(args.test_LR_path))
    val_HR = sorted(os.listdir(args.test_GT_path))
    
    Y_PSNR_list = []
    Y_SSIM_list = []
    
    if args.in_memory:

        for i in range(len(val_LR)):
            img_LR = Image.open(os.path.join(args.test_LR_path, val_LR[i]))
            img_LR = np.array(img_LR)
            
            if len(img_LR.shape) == 2:
                img_LR = np.expand_dims(img_LR, 2)
                img_LR = np.concatenate([img_LR] * args.channel, 2)
            
            img_HR = np.array(Image.open(os.path.join(args.test_GT_path, val_HR[i])))
            
            if args.self_ensemble:
                output = util.self_ensemble(args, model, sess, img_LR, is_recursive = args.chop_forward)
            
            else:
                if args.chop_forward:
                    output = util.recursive_forwarding(img_LR, args.scale, args.chop_size, sess, model, args.chop_shave)
                    output = output[0]
                else:
                    img_LR = np.expand_dims(img_LR, axis = 0)
                    output = sess.run(model.output, feed_dict = {model.LR : img_LR})
                    output = output[0]
                    
            
            h, w, c = output.shape
            val_gt = img_HR[:h,:w]
            
            if len(img_HR.shape) == 3:
                y_output = skimage.color.rgb2ycbcr(output)
                y_gt = skimage.color.rgb2ycbcr(val_gt)
                y_output = y_output / 255.0
                y_gt = y_gt / 255.0
                y_psnr = skimage.measure.compare_psnr(y_output[args.scale:-args.scale, args.scale:-args.scale, :1], y_gt[args.scale:-args.scale, args.scale:-args.scale, :1], data_range = 1.0)
                y_ssim = util.compare_ssim(y_output[args.scale:-args.scale, args.scale:-args.scale, 0], y_gt[args.scale:-args.scale, args.scale:-args.scale, 0], gaussian_weights=True, use_sample_covariance=False, data_range = 1.0)
                
            else:
                gray_output = skimage.color.rgb2gray(output)
                val_gt = val_gt / 255.0
                y_psnr = skimage.measure.compare_psnr(gray_output[args.scale:-args.scale, args.scale:-args.scale], val_gt[args.scale:-args.scale, args.scale:-args.scale], data_range = 1.0)
                y_ssim = util.compare_ssim(gray_output[args.scale:-args.scale, args.scale:-args.scale], val_gt[args.scale:-args.scale, args.scale:-args.scale], gaussian_weights=True, use_sample_covariance=False, data_range = 1.0)
                
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
            
            if len(img_HR.shape) == 3:
                y_output = skimage.color.rgb2ycbcr(output)
                y_gt = skimage.color.rgb2ycbcr(val_gt)
                y_output = y_output / 255.0
                y_gt = y_gt / 255.0
                
                y_psnr = skimage.measure.compare_psnr(y_output[args.scale:-args.scale, args.scale:-args.scale, :1], y_gt[args.scale:-args.scale, args.scale:-args.scale, :1], data_range = 1.0)
                y_ssim = util.compare_ssim(y_output[args.scale:-args.scale, args.scale:-args.scale, 0], y_gt[args.scale:-args.scale, args.scale:-args.scale, 0], gaussian_weights=True, use_sample_covariance=False, data_range = 1.0)
                
            else:
                gray_output = skimage.color.rgb2gray(output)
                val_gt = val_gt / 255.0
                
                y_psnr = skimage.measure.compare_psnr(gray_output[args.scale:-args.scale, args.scale:-args.scale], val_gt[args.scale:-args.scale, args.scale:-args.scale], data_range = 1.0)
                y_ssim = util.compare_ssim(gray_output[args.scale:-args.scale, args.scale:-args.scale], val_gt[args.scale:-args.scale, args.scale:-args.scale], gaussian_weights=True, use_sample_covariance=False, data_range = 1.0)
                
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
        
def test_only(args, model, sess, saver):
    
    saver.restore(sess,args.pre_trained_model)
    print("saved model is loaded for test only!")
    print("model path is %s"%args.pre_trained_model)
    
    val_LR = sorted(os.listdir(args.test_LR_path))
    
    if args.in_memory:

        for i in range(len(val_LR)):
            img_LR = Image.open(os.path.join(args.test_LR_path, val_LR[i])).convert("RGB")
            img_LR = np.array(img_LR)
            
            if args.self_ensemble:
                output = util.self_ensemble(args, model, sess, img_LR, is_recursive = args.chop_forward)
            
            else:
                if args.chop_forward:
                    output = util.recursive_forwarding(img_LR, args.scale, args.chop_size, sess, model, args.chop_shave)
                    output = output[0]
                else:
                    img_LR = np.expand_dims(img_LR, axis = 0)
                    output = sess.run(model.output, feed_dict = {model.LR : img_LR})
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

