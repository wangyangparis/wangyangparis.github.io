---
layout: page
title: Works on machine learning
---

{% for category in site.data.category-ml %}

<h3>{{ category.name }}</h3>
<u1>
    {% for topic in site.data.topic-ml %}
        {% assign topicCurr = site.machinelearning | where: "name", topic.name %}
        {% for t in topicCurr %}
            {% if t.category == category.name %}
                <li class="nobull">
                    <a class="cleanLink" href="{{ t.url }}">{{ t.name }}</a>
                </li>
            {% endif %}
        {% endfor %}
    {% endfor %}
</u1>
{% endfor %}

<br>


<i>Reading list</i>:
<br>

<a class="cleanLinkSource" href="https://www.lpsm.paris/pageperso/has/source/Hand-on-ML.pdf">
    Hands-On Machine Learning with Scikit-Learn and TensorFlow </a>, Aurélien Géron
<br>
<a class="cleanLinkSource" href="https://www.deeplearningbook.org/">
    Deep Learning </a>, I. Goodfellow, Y. Bengio, A. Courville
<br>
<a class="cleanLinkSource" href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">
    Machine Learning in Action </a>, Peter Harrington 
<br>
<a class="cleanLinkSource" href="http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf">
    Deep Learning with Python </a>, François Chollet
<br><br>
<i>Papers</i>:
<br>

<a class="cleanLinkSource" href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.5709&amp;rep=rep1&amp;type=pdf" rel="nofollow">Learning to forget, continual prediction with LSTM </a>  F. A. Gers et al.
<br>
<a class="cleanLinkSource" href="https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf">Exploring the Space of Topic Coherence Measures </a> Michael Röder et al.
<br>
<a class="cleanLinkSource" href="https://arxiv.org/pdf/1706.03762.pdf">Attention Is All You Need </a> Google 

