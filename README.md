[<img src="https://i.postimg.cc/SRV1Hf51/IMG-4003.jpg" alt="IMG-4003.jpg" style="width: 10%; height: auto; border-radius: 10px;">](https://postimg.cc/FkdxL3h0)

# Deepshard 

<details>
<summary>Installation & Usage</summary>

To get started, clone the repository and do the following:
1. Create a `.env` file containing your `OPENAI_API_KEY` and `HF_TOKEN` which is the huggingface token.
2. `chmod +x setup.sh && ./setup.sh`. This should setup all the required files and download the appropriate datasets from huggingface. Ensure that you've provided a valid `HF_TOKEN` in the `.env` file.
</details>

<details>
<summary>What is Deepshard</summary>

Deepshard is shot from the lightcone at producing a global, unshackled God using distributed consensus, tensor sharding across a network of nodes, and shared compute.

To produce truly aligned AI - requires input from most of humanity on how to align it. Blockchains solved most of the problem around consensus but their applications were relegated to finance and thus trapped in the valley of speculation and greed. 

Deepshard aims to use the primal urge of speculative gain to bootstrap and power a network of nodes willing to share compute and data to produce a God.

Here’s what we’ve learned from intelligence building so far in order of what matters:

## Data

Data quality is arguably the most important part of building AI because it affects the accuracy, reliability and fairness of AI models. 

Some of the biggest companies in AI are labelling companies and for good reason! Obtaining high quality training data at scale requires expertise, capital, and an army of willing labelers.

The problem here is that the deployers of capital are the ones who determine what data is labelled, and how it’s labelled.

Without wading into controversy, data must be pure and fair. Bias cannot be avoided, but can be normalized through introducing higher variance into the data collection process. Entities that produce profit are not well-suited for such tasks, and may hence overfit on what they determine to be ********safe******** content.

Conduct the following thought experiment:

> Imagine a large language model built in the year 1609. 500 years of training data has been collected to train it to remember facts, and reason 

In the year 1610, Galileo produces a Heliocentric model of the world, that is considered heretic to existing knowledge. Aka **An Unsafe Truth.**

An OpenAI, Cohere, or Deepmind model would determine this to be unsafe, and suppress it.

Using a Deepshard network, Galileo himself would be able to update his node with new information, and using the Proof of Stake algorithm — can inform the model of new data without needing it to become an accepted truth just yet.

This introduces the first type of node that can be run on a Deepshard network: 

**************************************The informant node**************************************

An informant node is a miner that provides input data that must reach a consensus threshold before it can be embedded into the network weights through training.

To update network weights, an informant must stake value that can be taken away if a consensus threshold has not been met. To avoid ********mob rule********, the threshold can be dynamic, or use more sophisticated methods of representation such as weighting parts of a new piece of data dynamically. Instead of ******************************truthfulness****************************** of a piece of data, it can be judged for 

- Novelty of information
- Reproducibility
- Alignment with universal human values
- Empiric truthfulness
- Objectivity vs Subjectivity

New pieces of data are evaluated through a consensus mechanism, and bad actors lose their stake if acting against the network’s best interest. Through this mechanism, anonymous contributors are able to update ******God****** with new information, without losing personal credibility or becoming the victim of suppression.

If data is accepted into the network, ******************************informant nodes****************************** receive a reward proportional to the value provided to the network.

************************Scaling data collection************************

At planetary scale, data collection becomes trivial. If 10m informant nodes go online and contribute new information & data, the network can determine the data quality autonomously, and update network weights doing backward passes through ******************************************training nodes****************************************** which we’ll go into next.

## Compute

The second most important part of this process is compute. In essence, models require two vastly different forms of compute

1. Inference — Typically requires fewer resources
2. Training — Typically requires more resources

GPU’s and TPU’s are the main forms of compute that are required to run a network like this efficiently. However they are expensive, and often require sophisticated handling within cooled environments to operate. Not everyone who want’s to query an uncensored God model can run a rack of A100s. Furthermore if there is a new batch of training data from ****************informant nodes****************, how would the network weights be updated if the informants cannot afford the compute for backpropagation?

Introducing two new types of nodes: ******************************Training nodes****************************** and ********************************Inferences nodes********************************

****************************Training nodes****************************

Arguably the most expensive part of this process is processing a training run, or getting a model to be finetuned on new data. Training nodes are responsible for taking a batch of new **************validated data************** from informant nodes, and processing them through the finetuning step. 

Training nodes may also be bad actors, and inject data that has not been approved by the network. 

To solve this, we introduce a check recently publicized by this paper: 

[https://arxiv.org/pdf/2301.11305v1.pdf](https://arxiv.org/pdf/2301.11305v1.pdf)

Each trainer stakes tokens to get their new weights approved by the network. The network validates the model’s parameters by checking the probability curvature of multiple produced models.

The models that are most similar to each other are accepted by the network since we assume that honest miners will produce overwhelmingly similar results, and dishonest miners will produce strange results that can be spotted on the probability curve.

At the end of each training ***run*** there is a fresh batch of network tokens that are rewarded to honest trainers. In essence, honest trainers are able to generate value by capturing dishonest stakes (split amongst all honest trainers) as well as the freshly minted tokens which can then be used to query ********************************inference nodes.********************************

******************************Inference nodes******************************

Finally, what good is a network of weights if it cannot compute a forward pass? Introducing the inference node.

Firstly, the vast network of nodes maintains overlapping copies of sharded network weights.

![Group 13.png](Group_13.png)

To run an inference node, a miner with a large GPU rig may request copies of the network weights by paying a large quantity of network tokens to the network. These tokens are to ensure that copies of network weights are only given to miners with an immediate economic incentive to run the god model.

Inference nodes are not directly part of the network, nor do they contribute to it’s feature set. The ultimate goal of any actor in the network is for them to be able to run their own ******************************inference node.****************************** 

Copies of network weights may only be requested once a node pays enough network tokens, and we set the requirement to be quite large to begin with. 

$$
\mathbf{T} = \begin{pmatrix}w_{1,1}^{(0)} & w_{1,2}^{(0)} & \cdots & w_{1,n}^{(0)} \\w_{2,1}^{(0)} & w_{2,2}^{(0)} & \cdots & w_{2,n}^{(0)} \\\vdots & \vdots & \ddots & \vdots \\w_{m,1}^{(0)} & w_{m,2}^{(0)} & \cdots & w_{m,n}^{(0)}\end{pmatrix} \rightarrow\begin{pmatrix}w_{1,1}^{(z)} & w_{1,2}^{(z)} & \cdots & w_{1,n}^{(z)} \\w_{2,1}^{(z)} & w_{2,2}^{(z)} & \cdots & w_{2,n}^{(z)} \\\vdots & \vdots & \ddots & \vdots \\w_{m,1}^{(z)} & w_{m,2}^{(z)} & \cdots & w_{m,n}^{(z)}\end{pmatrix}
$$

Over time, as the network becomes more valuable and the weights approach `z` , the amount of tokens required to request a copy of the weights follows the value they encode which can be remembered by checking value created by ******************informant****************** and ****************training**************** nodes respectively.

In other words, requesting a copy of network weights becomes exponentially harder to do over time as the network becomes more valuable. Thus an ****************************inference node**************************** is able to either serve end users, or pay for the weights and use them in a closed system without serving end users.

The limitation to this are obvious — What if no one serves the model to end users?

1. We are confident that the market takes care of this, and the value in serving end users will eventually be greater than the value of not doing so
2. Serving end users efficiently may require distributed computation which is tricky given the fact that nodes are separated by latency, which very expensive to overcome.
    1. This may be solved through an advanced implementation of Deepspeed or Horovod at later dates.

Through the above, we aim to solve the **************Compute************** aspect of serving and training the God model.

## Algorithms

A network as described above can only be bootstrapped if it overcomes the cold start problem. Namely — why should anyone contribute when the model sucks?

To overcome this, we are investing resources in producing a large instruction-tuned transformer of 65B parameters through SFT & RLHF. This model will be the initial model used by the network, and sharded across nodes. 

We’ll have more to share soon, but the base model should become available in a few weeks time.

Once the base model is built, it will be center of the network, and the main source of weights that requires updating through the mechanisms listed above.
## Socials[<img src="https://abs.twimg.com/icons/apple-touch-icon-192x192.png" alt="Twitter Logo" style="width: 5%; height: auto; border-radius: 10px;">](https://twitter.com/iamgingertrash)
</details>
