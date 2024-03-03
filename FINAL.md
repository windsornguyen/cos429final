# COS 429 Final Project

## Key Dates

- Milestone reports are due Wednesday, April 3rd @ 8:59 PM.
- Final reports are due on Dean's Date, May 7th.

**There are no extensions and no late submissions for any part of the final project.**

## Overview

The final assignment for this semester is to do an in-depth project implementing a nontrivial vision system. You will be expected to design a complete pipeline, read up on the relevant literature, implement the system, and evaluate it on real-world data. The project is worth 20% of your grade and will be graded out of 20 points, with the breakdown below. You will work individually or in small groups (2-3 people), and must deliver:

- A short (1-2 pages) milestone report

  - **Submit**: The milestone is due Wed, April 3rd 8:59pm.

  - **Details**: Submit one report per team through Gradescope. You will fill out text boxes with a description of the problem, an outline of the proposed approach, pointers to related course topics, plans for acquiring the necessary data/computational resources (see e.g., "Resources for Neural Networks"), plans for quantitative and qualitative evaluation, the target outcome (what do you expect to deliver at the end of the project?) and a fallback plan (what are the potential roadblocks? what is the minimum you will be able to deliver if the exploratory parts of the project go wrong?).

  - **The more detailed** your milestone writeup is, the more the course staff will be able to give concrete and useful feedback. Feel free to include brief questions as well if you have specific concerns about your proposal.

  - **Every team** will be assigned a “project advisor” from among the TAs who will serve as a resource in shaping the final project.

  - **Don't worry** if you end up changing course after the milestone -- this is expected, and there's no need to re-submit, just incorporate whatever feedback you receive into the final project.

- **Grading**: The milestone is worth 2 points (2% of the final course grade). Milestone grading will be straightforward: either full credit (all questions above answered thoughtfully), half credit (many questions left unanswered, or answers are very short) or zero credit (milestone not turned in by the final deadline, or equivalent).

- A final report on your project

  - **Submit**: The report is due **May 7th (Dean's Date)**. Please submit _one report per team_ as a PDF on Gradescope

    - In addition, **_please submit your code_** (or links to sites from which you downloaded pre-trained models, etc.), and links to any datasets you used. Unlike with the homework assignments, we will not be running your code but it does help us accurately assess and grade the scope of your implementation if we have questions beyond your report.

  - **Details**: The report should include sections on previous work, design and implementation, results, and a discussion of the strengths and weaknesses of your system. Include lots of pretty pictures! If you captured your own data, it is not necessary to submit a full dataset - just include a few samples. Even though you're submitting code, note that the grading will be based predominantly on the report. Thus, you need to carefully describe in your writeup what you implemented yourselves, along with any challenges you overcame in the implementation. You must also explicitly acknowledge all code, publications, and ideas you got from others outside your group.

  - **Grading**: The project report is worth 18 points (18% of the final course grade). You will be evaluated on the scope and success of your implementation, the rigor and depth of your scientific analysis, and the quality of your writeup.

    - _Implementation and analysis_ will be given equal consideration and will together comprise the majority of the report grade.

    - _The implementation grade_ will include both the scope of the system you tackled and the results you were able to get. They are graded together since frequently there’s a tradeoff: some of the deep learning systems may get better results than the classical systems but be straight-forward to implement and difficult to improve upon. Some of the more complex systems may be difficult to tune and thus while the implementation is complex the results will be poor. We will consider the quality of the results in the context of the complexity of the system.

    - The _analysis portion_ of the grade will include the motivation of your work and the quantitative and qualitative analysis. Think of your project first and foremost as a scientific exploration: What are your goals? Why did you make the design choices that you made? What did you think would happen as a result? Is this in fact what happened? If the results did not align with your intuition, why was that? In the end, what did your system do well? Where did it fail? Why? What are the steps to improve it? What have you learned about computer vision along the way?

    - Note that getting a high grade on the implementation and analysis portions implicitly relies on clear writing — if we can’t understand what you did, we can’t give you credit for it.

    - The _writeup_ grade will focus on the quality of the writeup beyond just explaining what you did — it will include the depth of discussion of related work, the quality of your figures and the organization of the report.

## FAQs

**Can I do a final project related to an IW or another class?**

Yes. If you're doing a related independent work project or a joint project between COS 429 and another class, in both the milestone and the final report you **must** additionally (1) describe the project you're doing outside COS 429, and (2) clearly articulate the component that's exclusive to COS 429.

**What is the most common pitfall for final projects?**

Every project **must** include both quantitative and qualitative evaluation; the most common pitfall is forgetting to include one or the other. The first question you want to ask yourself when thinking of a project is always “how will you know if the proposed system succeeds or fails?” Before diving into the implementation, consider exactly how you will go about evaluating your system. If you're unsure of how to evaluate your method, talk to the course staff.

**Do you have any advice? How do I get started?**

The best advice is to start early, define a metric of success, and build a baseline system as quickly as possible. What is the simplest pipeline you can build that goes from an input (image, video, RGBD image, …), performs the target task, produces an output, and is evaluated? Once you have that, you can begin improving the system, documenting the scientific exploration: evidence-driven hypothesis, implementation of the experiment, evaluation of the outcome, refined evidence-driven hypothesis, repeat.

**How do I know if my results are significant? What is a “good” result?**

First, replicating “state of the art” numbers is often very challenging and requires a lot of “secret sauce” and engineering tricks that may be beyond the scope of the project. We suggest not focusing on obtaining the best numbers overall but rather spending much of your time diving into qualitative and quantitative analysis, seeking to better understand and improve upon your baselines (which should hopefully be up and running by now). Your report needs to have not only the first baseline results but also scientific hypotheses and iterations. For establishing if your improvements over the baselines are significant, one simple technique is using [bootstrapping](http://host.robots.ox.ac.uk/pascal/VOC/pubs/bootstrap_note.pdf).

## Project scope

These projects are very flexible and adaptable to your interests/goals:

- you are free to focus on the topic(s) that excite you the most (you are even welcome to explore a computer vision topic outside the scope of the class),

- you can decide whether you want to collect your own visual data or use one of the existing benchmarks,

- you can build off of an existing toolbox or develop an algorithm entirely from scratch,

- you can focus your efforts more on analysis or more on building the system (although you should have some of both analysis and system building in your project).

Teams with 3 people are expected to do projects that are more somewhat more ambitious in scope than teams with 2 people. Feel free to confirm with the course staff if you're unsure.

_Project example:_

Suppose you select the topic of generic object detection, decide to use the standard benchmark dataset of [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and want to build off of an existing [Deformable Parts Model](https://cs.brown.edu/~pff/latent-release4/) toolbox. You then could:

1. Download the dataset and the software, and run the object detection system. You may or may not need to train the model (sometimes you can get access to pretrained models). Evaluate the results.

2. Use visualization or analysis techniques for understanding the errors in this system: this handy toolLinks to an external site. is great for the task of object detection in particular, but you can also use simpler techniques like confusion matrices or visualization of top-scoring images. Draw some conclusions of when the algorithm is succeeding and failing.

3. Identify one (or more key) parameters of the system: e.g., the number of deformable parts or the non-maximum suppression threshold. Evaluate how the results change, both quantitatively and qualitatively, as you vary these hyperparameters. Teams of 3 can challenge themselves to go deeper in this exploration: e.g., analyzing parameters that are inherent to how the model is trained, or exploring more of the parameters. How are the results changing as a function of these parameters? Is that consistent with your intuition?

4. Based on your exploration, formulate one concrete hypothesis for how to improve the object system. For example, perhaps adding global image context can improve object detection accuracy? Implement a way to verify your hypothesis. Evaluate how the results change quantitatively and qualitatively. Is your system better now? Teams of 3 can challenge themselves to go deeper, e.g., by exploring several avenues for improvement.

5. In the project report:

- Present your topic. Why is it important, e.g., what societal applications would benefit from improved object detection? What are the challenges in building a perfect object detector? Include pictures to illustrate the challenges.

- Describe the dataset: number of images, number of object classes, any interesting properties of the dataset. Show some example images. Don't forget to present the evaluation metric.

- Explain the DPM algorithm to the reader, as you would if you were teaching it in a COS 429 lecture.

- Present your analysis, including any hypotheses, intuitions or surprises, backed by both quantitative and qualitative results. This is the core of your work: make sure the reader walks away with a much more in-depth understanding of the challenge of object detection as a field and of the strengths and weaknesses of the DPM system in particular.

- Describe your modification(s) to the method, and the resulting quantitative and qualitative changes. If the modification(s) did not improve the method as expected, discuss some reasons for why this might be the case.- Acknowledge all code, publications, and ideas you got from others outside your group.

## Project ideas

You may select any computer vision topic that is of interest to you, but some ideas to get you started:

- Image mosaicing, including automatic image alignment and multiresolution blending.

- Foliage/tourist removal from several photos of a building. An important question to answer is whether you want to attempt 3D reconstruction as part of the process, or whether you want to consider it as a purely 2D problem.

- Video textures - see the SIGGRAPH paper linked from the [video textures web page](http://www.cc.gatech.edu/cpl/projects/videotexture/).

- Foreground/background segmentation (e.g., using the [Weizmann Horses](https://www.msri.org/people/members/eranb/) dataset)

- Any number of image recognition tasks:

  - OCR or handwriting recognition (e.g., using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset),

  - classifying images of skin rashes,

  - object classification (e.g., using the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) or [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) datasets),

  - object detection/semantic segmentation/human pose
    estimation/occlusion detection (e.g., check out the diverse [PASCAL VOC annotations](https://sites.google.com/view/pasd/dataset)),

  - object attributes (e.g., using [aPascal/aYahoo](http://vision.cs.uiuc.edu/attributes/) annotations or [ImageNet attributes](http://image-net.org/download-attributes)),

  - or even explore the interplay between different recognition tasks: object classification and attribute prediction, human pose estimation and action recognition, part segmentation and object detection, face detection and whole-person detection, etc.

- Explore and analyze the similarities and difference between different datasets and algorithms (e.g., check out the [Dataset Bias](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf). paper or the [ImageNet analysis (section 3)](http://ai.stanford.edu/~olga/papers/iccv13-ILSVRCanalysis.pdf)) -- your analysis should lead to at least one hypothesis that you verify experimentally.

- Develop an image captioning system combining existing recognition modules.

- Human action recognition in video (e.g., using the [KTH](http://www.nada.kth.se/cvap/actions/) dataset).

- Detect pose outliers in videos of dance performances, e.g., understand where performers deviate from the choreography.

- Pick your favorite computer vision algorithm, implement from scratch based only on the relevant publications (without looking at the reference implementation, if one exists), and analyze its accuracy, efficiency, sensitivity to different parameters, etc.

_Project ideas for those with graphics experience:_

- Inserting computer-generated objects into a video sequence taken with a moving camera. Use a calibration or structure from motion method to recover the camera pose.

- Some variant of Facade (human-assisted architectural modeling from a small number of photographs). See the the SIGGRAPH 96 paper linked from the [Facade web page](http://www.debevec.org/Research/).

- Vision-based automatic image morphing (e.g., of faces). That is, you use an optical flow or other correspondence method to generate matches between images, then use a morphing algorithm to generate intermediate frames.

- Image-based visual hull (shape from silhouettes) for moving scenes. See the SIGGRAPH 2000 paper, linked from their [web page](https://www.sci.utah.edu/~gerig/CS6320-S2015/Materials/ibvh-im-based-vis-hull-matusik.pdf).

## Past projects

For additional inspiration, consider the [outstanding projects from Fall 2017](http://www.cs.princeton.edu/courses/archive/fall17/cos429/featuredprojects.html).
