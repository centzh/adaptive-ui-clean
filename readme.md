# Developing a Vision-Language-Action Model to Drive Contextual User Interface Adaptation

This project investigates methods for **context-aware adaptation of mixed reality (MR) user interfaces**, with a focus on improving robustness and generalisation in complex and uncertain environments. The objective is to design UI systems that respond dynamically to the current context of use, optimising user experience across a wide range of real-world scenarios.

## Introduction

User interfaces in MR systems must account for a variety of contextual factors that can significantly influence usability. These include:

- **The structure and appearance of the physical environment**, which interfaces must integrate with and augment
- **The specific task** the user is engaged in, which determines the relevance and timing of interface elements
- **The user’s own physical and cognitive capabilities**, which constrain feasible interactions.

Each of these dimensions introduces **contextual uncertainty**. Traditional, hand-crafted adaptation strategies often fail under such uncertainty, particularly in **out-of-distribution** scenarios. This project aims to address this challenge by leveraging data-driven methods capable of generalising beyond the training distribution.

## Objectives

- Model context-sensitive features from egocentric input (e.g. visual appearance, user motion, task structure)
- Develop UI adaptation mechanisms that personalise interface behaviour in real time
- Investigate robustness to unseen environments, tasks, and user profiles
- Support continual adaptation and deployment in dynamic settings.

## Dataset

The dataset, containing visibility and placement task instances can be found [here](https://universityofcambridgecloud-my.sharepoint.com/:f:/g/personal/vz237_cam_ac_uk/Eh1sflOjWNBKnzDRQaPR9FIBiB2EX_btzbTV6EZP4WQj8w?e=JPlvKR).

## Installation

```bash
git clone https://github.com/centzh/adaptive-ui-clean.git
cd adaptive-ui
conda env create -f environment.yml
conda activate adaptive-ui```






 
