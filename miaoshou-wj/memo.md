- 使用[rank-bm25](https://pypi.org/project/rank-bm25/) / [bm25s ](https://huggingface.co/blog/xhluca/bm25s)来创建bm25搜索，qdrant自带的bm25不支持中文 bm25适合长文本
- qdrant的[bm42](https://blog.csdn.net/stephen147/article/details/140377467)（IDF+attention）适合短文本
- [bge-m3]( https://www.cnblogs.com/xiaoqi/p/18143552/bge-m3) 多语言性（Multi-Linguality）、多功能性（Multi-Functionality）和多粒度性（Multi-Granularity）

