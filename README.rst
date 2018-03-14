Introduction
============
|pypi peoject version|
|pypi pyversion|
|pypi license|
|travis ci|
|covergae|

JAQS是一个开源量化策略研究平台，由交易专家和金融技术专家共同设计，实现了自动化信号研究、高效策略开发和多维度回测分析，支持Alpha、CTA、套利等策略的实现。JAQS从实战而来，经实盘检验，本地化开发部署，保障策略安全。

|jaqsflowchart|

Features
========

- 通过统一的DataApi，获取、存储和管理数据。
- 通过数学公式快速定义并分析信号；实现Alpha选股、CTA、套利等各类量化交易策略，对策略进行历史回测。
- 通过统一的TradeApi，接入在线仿真系统进行仿真交易，跟踪策略表现。对接实盘通道实现实盘交易（当然需要用户搞定交易通道）。
- 完全本地化，代码可以部署在任意个人电脑或服务器上，本地化运行，策略安全性有保证。
- 模块化设计，通过标准的输入输出接口，做到数据与回测分离，交易与分析分离， 每一个环节都清晰可控，达到机构级别的标准化、流程化。
- 面向实盘编程，数据构建时进行严格的对齐，回测时提供当前快照而不是数据查询接口，防止未来函数的出现；通过对策略类的精巧设计，使回测与实盘/仿真交易可使用同一套策略代码，始于开展严谨的研究、回测。

Installation
============

参见 \ `安装指南 <https://github.com/quantOS-org/JAQS/blob/master/doc/install.md>`__\

**特别注意**

1. jaqs-大鱼金融fork版　在官方项目的基础上做了大量细节的优化和功能改动.在根据上述安装指南安装好官方jaqs相关版本依赖后,可以通过以下命令安装
 | pip install git+https://github.com/xingetouzi/JAQS.git@fxdayu
2. 如果事先已安装过官方版本的jaqs,需要pip uninstall jaqs卸载官方版本,再执行1中的命令安装大鱼金融fork版
3. 目前jaqs-大鱼金融fork版不支持直接pip更新,请随时关注fork版项目的更新进度,通过卸载旧版本并重新安装更新

What's new
==========

参见 \ `文档 <https://github.com/quantOS-org/JAQS/blob/master/doc/install.md>`__\

Document
========

参见 \ `文档 <https://github.com/quantOS-org/JAQS/blob/master/doc/install.md>`__\


Questions
==========
- 查看jaqs官方项目,请到\ `这里 <https://github.com/quantOS-org/JAQS>`__\。
- 访问大鱼金融官网获得更多使用案例和金融量化知识(\ `大鱼金融官网 <http://www.fxdayu.com>`__\)。
- 加qq群(372592121)进行讨论


License
=======

Apache 2.0许可协议。



.. |jaqsflowchart| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/jaqs.png

.. |pypi peoject version| image:: https://img.shields.io/pypi/v/jaqs.svg
   :target: https://pypi.python.org/pypi/jaqs
.. |pypi license| image:: https://img.shields.io/pypi/l/jaqs.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |pypi pyversion| image:: https://img.shields.io/pypi/pyversions/jaqs.svg
   :target: https://pypi.python.org/pypi/jaqs
.. |travis ci| image:: https://travis-ci.org/quantOS-org/JAQS.svg?branch=master
   :target: https://travis-ci.org/quantOS-org/JAQS
.. |covergae| image:: https://coveralls.io/repos/github/quantOS-org/JAQS/badge.svg?branch=master
   :target: https://coveralls.io/github/quantOS-org/JAQS?branch=master
