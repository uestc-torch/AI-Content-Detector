
这是我们在“炬火种·燃新薪”2024厦门火炬高新区高校专业新星挑战大赛,人工智能赛道-AI内容检测任务中实现的 __图注意力检索增强的两阶段AI生成内容检测方法__，
由于git大文件上传的限制，其中`pretrained_best`文件夹我们已上传至[百度网盘](https://pan.baidu.com/s/1_1UEOKBfp36iufE2q0ZCZg?pwd=8888)，请您下载后将其请放在主项目文件夹下。


<img width="946" alt="image" src="https://github.com/user-attachments/assets/da9f8325-4b83-4b08-b8b4-863604752b7c">
下表是我们在切分出的验证集上的评估效果:

| QW7B | RetrievalWithUnpretrainedModel | Pretrained Model | RetrievalWithPretrainedModel |
|------|--------------------------------|------------------|------------------------------|
| 53%  | 94.5%                          | 96%              | 98%                          |

测试时运行eval_with_retrival.py即可正常输出结果至evaluation_results.csv文件，同时模型也会输出eval_accuracy结果
