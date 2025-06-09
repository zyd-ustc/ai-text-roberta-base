import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import argparse
from tqdm import tqdm
import os

class AIGCDetector:
    def __init__(self, model_path):
        """
        初始化AIGC检测器
        
        Args:
            model_path: 模型文件夹路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        checkpoint = torch.load(pt_model_path, map_location=self.device)
        config = RobertaConfig.from_pretrained(tokenizer_path)
        self.model = RobertaForSequenceClassification(config)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，来自: {model_path}")
    
    def predict_single(self, text, max_length=512):
        """
        对单个文本进行预测
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            
        Returns:
            dict: 包含预测结果的字典
        """
        # 文本预处理和tokenization
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_label': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'is_ai_generated': predicted_class == 1  # 假设1表示AI生成
        }
    
    def detect_jsonl(self, input_file, submit_file="submit.txt", detailed_output_file=None, batch_size=16, max_length=512):
        """
        检测jsonl文件中的文本
        
        Args:
            input_file: 输入jsonl文件路径
            submit_file: 提交文件路径（每行一个label）
            detailed_output_file: 详细结果文件路径（可选）
            batch_size: 批处理大小
            max_length: 最大序列长度
        """
        results = []
        labels = []  # 存储预测标签，用于submit.txt
        
        # 读取jsonl文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"开始处理 {len(lines)} 条数据...")
        
        # 批处理预测
        for i in tqdm(range(0, len(lines), batch_size), desc="检测进度"):
            batch_lines = lines[i:i+batch_size]
            batch_texts = []
            batch_data = []
            
            # 解析当前批次的数据
            for line in batch_lines:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    batch_texts.append(text)
                    batch_data.append(data)
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line[:50]}...")
                    # 为跳过的行添加默认标签0（人类文本）
                    labels.append(0)
                    continue
            
            if not batch_texts:
                continue
            
            # 批量tokenization
            inputs = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 批量预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
            
            # 处理结果
            for j, (data, pred_class, confidence, probs) in enumerate(
                zip(batch_data, predicted_classes, confidences, probabilities)
            ):
                pred_label = pred_class.item()
                labels.append(pred_label)  # 添加到标签列表
                
                result = {
                    **data,  # 保留原始数据
                    'predicted_label': pred_label,
                    'confidence': confidence.item(),
                    'probabilities': probs.cpu().numpy().tolist(),
                    'is_ai_generated': pred_label == 1
                }
                results.append(result)
        
        # 保存submit.txt文件（每行一个标签）
        with open(submit_file, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(f"{label}\n")
        print(f"提交文件已保存到: {submit_file}")
        
        # 保存详细结果（可选）
        if detailed_output_file:
            with open(detailed_output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"详细结果已保存到: {detailed_output_file}")
        
        # 统计信息
        ai_count = sum(1 for label in labels if label == 1)
        human_count = len(labels) - ai_count
        
        print(f"\n检测完成!")
        print(f"总计: {len(labels)} 条")
        print(f"AI生成: {ai_count} 条 ({ai_count/len(labels)*100:.1f}%)")
        print(f"人类写作: {human_count} 条 ({human_count/len(labels)*100:.1f}%)")
        
        return results, labels

def main():
    parser = argparse.ArgumentParser(description='AIGC文本检测器 - 检测jsonl格式数据')
    parser.add_argument('--model_path', type=str, 
                       default='/Users/yida/Documents/2025/深度学习导论/tianchi/AIGC_detector_enbeta',
                       help='模型文件夹路径')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入jsonl文件路径')
    parser.add_argument('--submit_file', type=str, default='submit.txt',
                       help='提交文件路径（每行一个label）')
    parser.add_argument('--detailed_output_file', type=str,
                       help='详细结果文件路径（可选）')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return
    
    try:
        # 初始化检测器
        detector = AIGCDetector(args.model_path)
        
        # 执行检测
        results, labels = detector.detect_jsonl(
            input_file=args.input_file,
            submit_file=args.submit_file,
            detailed_output_file=args.detailed_output_file,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # 显示一些示例结果
        print("\n示例结果:")
        for i, (result, label) in enumerate(zip(results[:3], labels[:3])):
            print(f"\n样本 {i+1}:")
            print(f"文本: {result.get('text', '')[:100]}...")
            print(f"预测标签: {label} ({'AI生成' if label == 1 else '人类写作'})")
            print(f"置信度: {result['confidence']:.3f}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()