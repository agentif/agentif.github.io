---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Home
---

<div>

<p align="center">
  👥 <a href="https://github.com/THU-KEG/AgentIF/" target="_blank" rel="noopener noreferrer nofollow">AgentIF Team</a> •
  📚 <a href="https://arxiv.org/abs/2505.16944" target="_blank" rel="noopener noreferrer nofollow">AgentIF Paper</a> •
  <img src="images/github.png" width=15> <a href="https://github.com/THU-KEG/AgentIF/" target="_blank" rel="noopener noreferrer nofollow">Code</a> •
  📊 <a href="https://huggingface.co/datasets/THU-KEG/AgentIF" target="_blank" rel="noopener noreferrer nofollow">AgentIF Dataset</a>
</p>

</div>

![Logo](images/logo.png)

---

We introduce **AgentIF**, the first benchmark for systematically **evaluating LLM instruction following ability in agentic scenarios**. AgentIF features three key characteristics: (1) **Realistic**, constructed from 50 real-world agentic applications. (2) **Long**, averaging 1,723 words with a maximum of 15,630 words. (3) **Complex**, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. Here is the instruction length distribution in AgentIF, along with the success rates of several representative LLMs across the constraint dimensions we propose:
![Logo](images/fig1.png)
An example instruction of AgentIF:
![Logo](images/example_1.png)

## Leaderboard

#### Metrics
- Constraint Success Rate (CSR) measures the proportion of individual constraints that are correctly satisfied by the model’s response. 
- Instruction Success Rate (ISR) measures the proportion of instructions for which all constraints are satisfied. 

#### Performance Across Constraint Categories

- Press each model to get its latest results. Or press this key to reach the depository of all results: 
<a href="https://github.com/agentif/agentif.github.io/tree/main/docs/results" target="_blank" rel="noopener noreferrer nofollow">Results</a>

<table>
  <thead>
    <tr>
      <td rowspan="2"><b>Models</b></td>
      <td colspan="3"><b>Dimension</b></td>
      <td colspan="3"><b>Type</b></td>
      <td rowspan="2"><b>ISR</b></td>
      <td rowspan="2"><b>CSR</b></td>
    </tr>
    <tr>
      <td><b>Vanilla</b></td>
      <td><b>Condition</b></td>
      <td><b>Example</b></td>
      <td><b>Formatting</b></td>
      <td><b>Semantic</b></td>
      <td><b>Tool</b></td>
    </tr>
  </thead>
  <tbody>
    <tr><td>[T] <a href="results/o1-mini_gpt4o.zip" download>o1-mini</a></td><td>59.8</td><td>37.5</td><td>80.8</td><td>66.1</td><td>59.1</td><td>43.2</td><td>26.9</td><td>59.8</td></tr>
    <tr><td>[N] <a href="results/gpt-4o-2024-11-20_gpt4o.zip" download>GPT-4o</a></td>
    <td>58.0</td><td>35.1</td><td>80.8</td><td>65.8</td><td>56.5</td><td>43.2</td><td>26.4</td><td>58.5</td></tr>
    <tr><td>[N] <a href="results/Qwen3-32B_gpt4o.zip" download>Qwen3-32B</a></td>
    <td>57.5</td><td>41.1</td><td>80.6</td><td>57.7</td><td>62.5</td><td>45.7</td><td>24.9</td><td>58.4</td></tr>
    <tr><td>[T] <a href="results/QwQ-32B_gpt4o.zip" download>QwQ-32B</a></td>
    <td>57.5</td><td>35.6</td><td>82.7</td><td>61.4</td><td>59.4</td><td>43.2</td><td>27.2</td><td>58.1</td></tr>
    <tr><td>[T] <a href="results/deepseek-r1_gpt4o.zip" download>DeepSeek-R1</a></td>
    <td>56.1</td><td>41.4</td><td>87.0</td><td>61.4</td><td>58.9</td><td>44.4</td><td>22.2</td><td>57.9</td></tr>
    <tr><td>[T] <a href="results/GLM-Z1-32B-0414_gpt4o.zip" download>GLM-Z1-32B</a></td>
    <td>56.7</td><td>37.9</td><td>83.6</td><td>60.2</td><td>59.6</td><td>43.1</td><td>23.8</td><td>57.8</td></tr>
    <tr><td>[N] <a href="results/deepseek-v3-250324_gpt4o.zip" download>DeepSeek-V3</a></td>
    <td>54.9</td><td>41.5</td><td>84.5</td><td>59.3</td><td>58.9</td><td>40.8</td><td>21.9</td><td>56.7</td></tr>
    <tr><td>[N] <a href="results/claude-3-5-sonnet-20241022_gpt4o.zip" download>Claude-3-5-Sonnet</a></td>
    <td>57.3</td><td>36.9</td><td>69.2</td><td>61.5</td><td>56.0</td><td>43.3</td><td>24.9</td><td>56.6</td></tr>
    <tr><td>[N] <a href="results/Meta-Llama-3.1-70B-Instruct_gpt4o.zip" download>Meta-Llama-3.1-70B-Instruct</a></td>
    <td>55.1</td><td>35.0</td><td>84.3</td><td>61.6</td><td>55.6</td><td>42.8</td><td>20.9</td><td>56.3</td></tr>
    <tr><td>[T] <a href="results/DeepSeek-R1-Distill-Qwen-32B_gpt4o.zip" download>DeepSeek-R1-Distill-Qwen-32B</a></td>
    <td>54.5</td><td>39.6</td><td>73.1</td><td>55.7</td><td>57.2</td><td>45.2</td><td>20.7</td><td>55.1</td></tr>
    <tr><td>[T] <a href="results/DeepSeek-R1-Distill-Llama-70B_gpt4o.zip" download>DeepSeek-R1-Distill-Llama-70B</a></td>
    <td>55.4</td><td>37.7</td><td>69.2</td><td>56.5</td><td>56.6</td><td>44.1</td><td>19.9</td><td>55.0</td></tr>
    <tr><td>[N] <a href="results/Meta-Llama-3.1-8B-Instruct_gpt4o.zip" download>Meta-Llama-3.1-8B-Instruct</a></td>
    <td>53.5</td><td>36.6</td><td>71.4</td><td>55.6</td><td>54.8</td><td>43.5</td><td>19.9</td><td>53.6</td></tr>
    <tr><td>[S] <a href="results/Mistral-Crab-DPO_gpt4o.zip" download>Crab-DPO-7B</a></td>
    <td>48.3</td><td>24.3</td><td>57.5</td><td>48.8</td><td>47.4</td><td>41.9</td><td>10.1</td><td>47.2</td></tr>
    <tr><td>[N] <a href="results/Mistral-7B-Instruct-v0.3_gpt4o.zip" download>Mistral-7B-Instruct-v0.3</a></td>
    <td>47.9</td><td>29.2</td><td>53.8</td><td>47.0</td><td>48.6</td><td>39.8</td><td>11.5</td><td>46.8</td></tr>
    <tr><td>[S] <a href="results/Conifer_dpo_gpt4o.zip" download>Conifer-DPO-7B</a></td>
    <td>45.6</td><td>27.0</td><td>50.5</td><td>42.0</td><td>46.9</td><td>41.8</td><td>10.7</td><td>44.3</td></tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="9" style="text-align: left; font-size: 0.9em; padding: 10px; border-top: 2px solid #ddd;">
	Success rates (%) of various proprietary and open-source LLMs on <span style="font-variant: small-caps;">AgentIF</span>, sorted by CSR in descending order. [N] denotes non-thinking models, [T] denotes thinking models, and [S] denotes models explicitly designed for instruction following by the academic community.
      </td>
    </tr>
  </tfoot>
</table>

## Evaluation
For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation.

### How to evaluation
1. Clone the remote repository to your local environment. The necessary data is already included, so no further actions are needed.
    ```
    git clone https://github.com/THU-KEG/AgentIF.git
    ```
    
2. (Optional) To evaluate a model hosted locally, deploy it using vLLM. Use a command similar to the following:
    ```bash
    CUDA_VISIBLE_DEVICES=<CUDA_ID> vllm serve "<your_model_path>" \
        --served-model-name <your_model_name> \
        --port 8008 \
        --tensor-parallel-size <num_gpus> \
        --max-model-len 32000 \
        --gpu-memory-utilization 0.9
    ```


2. Specify the target model and the evaluator in the `run.sh` file. To reproduce our results, we recommend using `gpt-4o-2024-11-20`.

   ```
   Model_Name=""             # Name of the model to evaluate
   Model_Name_URL=""         # Endpoint of the model (e.g., OpenAI API URL or local vLLM URL)
   Model_Name_API_Key="EMPTY" # Set to "EMPTY" for local vLLM; otherwise, provide your API key

   Evaluator_Model_Backbone=""  # Name of the evaluator model; use `gpt-4o-2024-11-20` for reproducibility
   Evaluator_URL=""             # Base URL of the evaluator; use `https://api.openai.com/v1` to match our setup
   Evaluator_API_Key=""         # API key for the evaluator
   ```
    
3. Then run the script to start the evaluation.
    
    ```
    sh run.sh
    ```

## Data Format
Each data instance in AgentIF is structured as follows:
```
{
  "input": [
    { "role": "system", "content": "..." },
    { "role": "user",   "content": "..." }
  ],
  "constraints": [
    {
      "id": 0,
      "desc": "...",                // Constraint description
      "other_info": {               // Auxiliary information for evaluation
        "...": "..."
      },
      "dimension": "...",           // Constraint Presentation Type
      "type": "...",                // Constraint Type
      "is_meta": false,             // Whether it is a meta-constraint
      "evaluation": [               // Evaluation Method
        {
          "type": "llm",            // LLM-based evaluation
          "required_keys": ["response"],
          "exec": "..."             // Evaluation prompt for LLM
        },
        {
          "type": "code",           // Code-based evaluation
          "exec": "..."             // Executable code snippet
        }
      ]
    }
  ]
}
```

## Citation

```
@misc{qi2025agentifbenchmarkinginstructionfollowing,
      title={AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios}, 
      author={Yunjia Qi and Hao Peng and Xiaozhi Wang and Amy Xin and Youfeng Liu and Bin Xu and Lei Hou and Juanzi Li},
      year={2025},
      eprint={2505.16944},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.16944}, 
}
```
