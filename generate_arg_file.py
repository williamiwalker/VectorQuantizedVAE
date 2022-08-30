# This creates a json file that has a list of arguments that each job in a slurm job array will read
import json
import os



################################
# VQVAE ARGUMENTS
################################
SUBFOLDER_NAME = 'VQVAE_3'
ARG_FILE_NAME = 'arguments_' + SUBFOLDER_NAME +'.json'
ARGUMENT_FILE = '/nfs/gatsbystor/williamw/svae/arg_files/'+ARG_FILE_NAME

COMMENTS = {'GSSOFT':'paired MNIST images data. Using gumbel softmax VAE',
            'VQVAE': 'paired MNIST images data. Using vector quantized VAE'
            }

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
#     parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume.")
#     parser.add_argument("--model", choices=["VQVAE", "GSSOFT"], help="Select model to train (either VQVAE or GSSOFT)")
#     parser.add_argument("--channels", type=int, default=256, help="Number of channels in conv layers.")
#     parser.add_argument("--latent-dim", type=int, default=8, help="Dimension of categorical latents.")
#     parser.add_argument("--num-embeddings", type=int, default=128, help="Number of codebook embeddings size.")
#     parser.add_argument("--embedding-dim", type=int, default=32, help="Dimension of codebook embeddings.")
#     parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
#     parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
#     parser.add_argument("--num-training-steps", type=int, default=250, help="Number of training steps.")
mainArgs = {
'MAIN_FOLDER': SUBFOLDER_NAME,
'num_workers': 4,
'resume': None,
'channels': 128,
'latent_dim': 1,
'num_embeddings': 10,
'embedding_dim': 64,
'learning_rate': 5e-4,
'batch_size': 16,
'num_epochs': 200
}

MODEL_TYPES = ['VQVAE','GSSOFT']




arguments = {}

job_index = 0
for indm, modelType in enumerate(MODEL_TYPES):
    currDict = mainArgs.copy()
    currDict['model'] = modelType
    currDict['SUB_FOLDER'] = modelType
    currDict['COMMENTS'] = COMMENTS[modelType]
    arguments[job_index] = currDict
    job_index += 1

print('sbatch --array=0-'+ str(job_index-1) + ' train_VQVAE.sbatch')





if os.path.exists(ARGUMENT_FILE):
    print('overwrite')
    # raise Exception('You tryina overwrite a folder that already exists. Dont waste my time, sucka')

with open(ARGUMENT_FILE, 'w') as f:
    json.dump(arguments, f, indent=4)
#

