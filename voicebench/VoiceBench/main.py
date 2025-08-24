from datasets import load_dataset, Audio
from argparse import ArgumentParser
from src.models import model_cls_mapping
import json
from tqdm import tqdm
from loguru import logger


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2', choices=list(model_cls_mapping.keys()))
    parser.add_argument('--data', type=str, default='alpacaeval')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'text', 'ttft'])
    args = parser.parse_args()

    # load data
    data = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    # load model
    model = model_cls_mapping[args.model]()
    # data = data.select([0,1,2,3,4,5])

    if args.modality == 'ttft':
        # avoid cold start
        _ = model.generate_ttft(data[0]['audio'])

    # inference
    results = []
    for item in tqdm(data, total=len(data)):
        tmp = {k: v for k, v in item.items() if k != 'audio'}
        if args.modality == 'text':
            response = model.generate_text(item['prompt'])
        elif args.modality == 'audio':
            response = model.generate_audio(item['audio'])
        elif args.modality == 'ttft':
            response = model.generate_ttft(item['audio'])
        else:
            raise NotImplementedError
        logger.info(item['prompt'])
        logger.info(response)
        logger.info('====================================')
        tmp['response'] = response
        results.append(tmp)

    # save results
    output_file = f'{args.model}-{args.data}-{args.split}-{args.modality}.jsonl'
    with open(output_file, 'w') as f:
        for record in results:
            json_line = json.dumps(record)  # Convert dictionary to JSON string
            f.write(json_line + '\n')


if __name__ == '__main__':
    main()
