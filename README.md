# 基于 Langchain 架构实现 LLM 从0到1的开发

本项目主要参考 [LangChain 官方文档](https://python.langchain.com/docs) 、[《Learning LangChain -- Mayo Oshin & Nuno Campos(青蛙书)](https://pan.baidu.com/s/1KgTvOfzzoHoLFxxS0Ie0gg)

本项目目前计划按照 LangChain 的几个特色模块**基础**、**memory**、**agents**、**tools**、**chains**、**prompt**、**output parser**、**RAG**、**LangGraph**，以具体代码实例和讲解的方式来实现。

## Setup
本项目运行在 python=3.12 ubuntu 22.04 环境下，依赖包在environment.yml中，可以通过以下命令安装：
```bash
conda env create -f --name your-conda-environment-name environment.yml 
```
此外，本项目是基于[阿里百炼平台](https://bailian.console.aliyun.com/#/home)所提供的  [qwen-plus](https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html) 模型，因此需要先申请模型密钥，并设置环境变量：
```bash
export BAILIAN_API_KEY=your-api-key
```
我一般的做法是在当前目录下人为生成一个.env文件，并写入BAILIAN_API_KEY=your-api-key，然后通过以下命令加载环境变量：
```bash
load_dotenv()
```
> 阿里百炼对新用户有100万 tokens 的免费额度，可以申请。此外[硅基流动](https://siliconflow.cn/zh-cn/models)、[魔搭社区](https://modelscope.cn/)等平台也提供了免费额度，可以申请。
