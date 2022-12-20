这个的依赖环境比较简单，除了 anaconda里面常带的，只需要额外再安装pytorch+opencv就可以。    

里面基本文件和文件夹分别为：    
  datasets： 示例数据集，使用运行示例代码运行本数据集结果为 58.766    
  models： 模型结构配置文件    
  params： 模型参数文件    
  heatmap_res_imgs： 示例代码自动生成的热力结果图    
  alpha_res_imgs:  示例代码自动生成的alpha渲染图   
  test.py： 示例代码。很容易读懂。    
  test_video.py: 针对Video的示例代码   
  如果仅仅是使用，则具备models+params+test.py就可以


 方法详述在：https://github.com/Zhaoyi-Yan/DCANet    

时间统计为:    
  在平均分辨率为[574.46153846 861.91208791]的图片上模型运行出结果的平均时间为94ms, 产生渲染图的平均时间为9ms, 所以总运行时间差不多为100ms. 测试显卡为NVIDIA Corporation Device 2208








