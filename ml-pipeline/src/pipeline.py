"""
Support Ticket Classification Pipeline

Complete end-to-end ML pipeline for support ticket classification and priority prediction.
Includes data generation, preprocessing, model training, and result export.
"""

import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import train_test_split

from data_generator import SupportTicketGenerator
from preprocessing import TextPreprocessor, FeatureExtractor
from model_trainer import ModelTrainer


class SupportTicketPipeline:
    """Complete ML pipeline for support ticket classification."""

    def __init__(self, output_dir: str = "../data"):
        """Initialize pipeline."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.tickets = []
        self.preprocessor = TextPreprocessor()
        self.trainer = ModelTrainer()

        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train_category = None
        self.y_test_category = None
        self.y_train_priority = None
        self.y_test_priority = None

        self.category_mapping = {}
        self.priority_mapping = {}
        self.results = {}

    def step_1_generate_data(self, num_tickets: int = 500) -> Dict[str, Any]:
        """
        Step 1: Generate synthetic support ticket dataset.

        Args:
            num_tickets: Number of tickets to generate

        Returns:
            Dataset statistics
        """
        print("\n" + "="*60)
        print("STEP 1: GENERATING SYNTHETIC DATASET")
        print("="*60)

        self.tickets, stats = SupportTicketGenerator.generate_dataset(num_tickets)

        # Save raw dataset
        dataset_path = os.path.join(self.output_dir, "support_tickets_raw.json")
        SupportTicketGenerator.save_dataset(self.tickets, dataset_path)

        print(f"\n✓ Generated {num_tickets} support tickets")
        print(f"  Saved to: {dataset_path}")
        print(f"\nDataset Statistics:")
        print(f"  Total Tickets: {stats['total_tickets']}")
        print(f"\n  Category Distribution:")
        for cat, count in stats['category_distribution'].items():
            pct = count / stats['total_tickets'] * 100
            print(f"    - {cat}: {count} ({pct:.1f}%)")
        print(f"\n  Priority Distribution:")
        for pri, count in stats['priority_distribution'].items():
            pct = count / stats['total_tickets'] * 100
            print(f"    - {pri}: {count} ({pct:.1f}%)")

        return stats

    def step_2_preprocess_data(self) -> Dict[str, Any]:
        """
        Step 2: Preprocess text data and extract features.

        Returns:
            Preprocessing statistics
        """
        print("\n" + "="*60)
        print("STEP 2: PREPROCESSING TEXT DATA")
        print("="*60)

        # Extract text and labels
        texts = [ticket['text'] for ticket in self.tickets]
        categories = [ticket['category'] for ticket in self.tickets]
        priorities = [ticket['priority'] for ticket in self.tickets]

        # Preprocess texts
        print("\n  Cleaning and tokenizing texts...")
        preprocessed_texts = [self.preprocessor.preprocess_text(text) for text in texts]

        # Calculate text statistics
        text_stats = TextPreprocessor.calculate_text_statistics(preprocessed_texts)
        print(f"\n  Text Statistics:")
        print(f"    - Total Documents: {text_stats['total_documents']}")
        print(f"    - Total Words: {text_stats['total_words']}")
        print(f"    - Unique Words: {text_stats['unique_words']}")
        print(f"    - Avg Words per Document: {text_stats['avg_words_per_doc']:.1f}")
        print(f"    - Word Range: {text_stats['min_words']} - {text_stats['max_words']}")

        # Encode labels
        print("\n  Encoding labels...")
        y_category, self.category_mapping = self.preprocessor.encode_labels(categories, "category")
        y_priority, self.priority_mapping = self.preprocessor.encode_labels(priorities, "priority")

        print(f"    - Categories: {list(self.category_mapping.values())}")
        print(f"    - Priorities: {list(self.priority_mapping.values())}")

        # TF-IDF vectorization
        print("\n  Extracting TF-IDF features...")
        X_tfidf = self.preprocessor.fit_transform_tfidf(preprocessed_texts, max_features=500)
        print(f"    - Feature Matrix Shape: {X_tfidf.shape}")

        # Extract additional features
        print("\n  Extracting hand-crafted features...")
        X_features = FeatureExtractor.extract_batch_features(texts)
        print(f"    - Hand-crafted Features Shape: {X_features.shape}")

        # Combine features
        X = np.hstack([X_tfidf, X_features])
        print(f"    - Combined Feature Matrix Shape: {X.shape}")

        # Split data
        print("\n  Splitting data (80% train, 20% test)...")
        (self.X_train, self.X_test,
         self.y_train_category, self.y_test_category,
         self.y_train_priority, self.y_test_priority) = train_test_split(
            X, y_category, y_priority,
            test_size=0.2,
            random_state=42,
            stratify=y_category
        )

        print(f"    - Training Set Size: {self.X_train.shape[0]}")
        print(f"    - Test Set Size: {self.X_test.shape[0]}")

        return {
            "text_statistics": text_stats,
            "feature_matrix_shape": X.shape,
            "train_size": self.X_train.shape[0],
            "test_size": self.X_test.shape[0]
        }

    def step_3_train_category_models(self) -> Dict[str, Any]:
        """
        Step 3: Train models for category classification.

        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("STEP 3: TRAINING CATEGORY CLASSIFICATION MODELS")
        print("="*60)

        # Create and train models
        self.trainer.create_models()
        print("\n  Training models...")
        self.trainer.train_models(self.X_train, self.y_train_category, verbose=True)

        # Evaluate models
        print("\n  Evaluating models...")
        category_results = self.trainer.evaluate_models(
            self.X_test, self.y_test_category,
            label_mapping=self.category_mapping,
            verbose=True
        )

        self.results['category'] = {
            'model_name': self.trainer.best_model_name,
            'results': category_results
        }

        return category_results

    def step_4_train_priority_models(self) -> Dict[str, Any]:
        """
        Step 4: Train models for priority prediction.

        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("STEP 4: TRAINING PRIORITY PREDICTION MODELS")
        print("="*60)

        # Reset trainer for priority prediction
        trainer_priority = ModelTrainer()
        trainer_priority.create_models()

        print("\n  Training models...")
        trainer_priority.train_models(self.X_train, self.y_train_priority, verbose=True)

        # Evaluate models
        print("\n  Evaluating models...")
        priority_results = trainer_priority.evaluate_models(
            self.X_test, self.y_test_priority,
            label_mapping=self.priority_mapping,
            verbose=True
        )

        self.results['priority'] = {
            'model_name': trainer_priority.best_model_name,
            'results': priority_results,
            'trainer': trainer_priority
        }

        return priority_results

    def step_5_generate_predictions(self) -> Dict[str, Any]:
        """
        Step 5: Generate predictions on test set.

        Returns:
            Predictions and analysis
        """
        print("\n" + "="*60)
        print("STEP 5: GENERATING PREDICTIONS")
        print("="*60)

        # Get predictions
        category_trainer = self.trainer
        priority_trainer = self.results['priority']['trainer']

        y_pred_category = category_trainer.predict(self.X_test)
        y_pred_priority = priority_trainer.predict(self.X_test)

        # Decode predictions
        pred_categories = self.preprocessor.decode_labels(y_pred_category, "category")
        pred_priorities = self.preprocessor.decode_labels(y_pred_priority, "priority")

        # Create prediction results
        predictions = []
        for i in range(len(self.X_test)):
            predictions.append({
                "ticket_index": i,
                "predicted_category": pred_categories[i],
                "actual_category": self.category_mapping[self.y_test_category[i]],
                "predicted_priority": pred_priorities[i],
                "actual_priority": self.priority_mapping[self.y_test_priority[i]],
                "category_correct": pred_categories[i] == self.category_mapping[self.y_test_category[i]],
                "priority_correct": pred_priorities[i] == self.priority_mapping[self.y_test_priority[i]]
            })

        print(f"\n✓ Generated predictions for {len(predictions)} test samples")
        print(f"  Category Accuracy: {sum(p['category_correct'] for p in predictions) / len(predictions) * 100:.2f}%")
        print(f"  Priority Accuracy: {sum(p['priority_correct'] for p in predictions) / len(predictions) * 100:.2f}%")

        return predictions

    def step_6_export_results(self, predictions: List[Dict]) -> None:
        """
        Step 6: Export all results to JSON files.

        Args:
            predictions: Prediction results
        """
        print("\n" + "="*60)
        print("STEP 6: EXPORTING RESULTS")
        print("="*60)

        # Export predictions
        predictions_path = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\n✓ Saved predictions to: {predictions_path}")

        # Export model results
        results_path = os.path.join(self.output_dir, "model_results.json")
        export_results = {}

        for task, task_data in self.results.items():
            export_results[task] = {
                'best_model': task_data['model_name'],
                'models': {}
            }

            for model_name, metrics in task_data['results'].items():
                export_results[task]['models'][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                }

        with open(results_path, 'w') as f:
            json.dump(export_results, f, indent=2)
        print(f"✓ Saved model results to: {results_path}")

        # Export mappings
        mappings_path = os.path.join(self.output_dir, "label_mappings.json")
        with open(mappings_path, 'w') as f:
            json.dump({
                'categories': self.category_mapping,
                'priorities': self.priority_mapping
            }, f, indent=2)
        print(f"✓ Saved label mappings to: {mappings_path}")

    def run_full_pipeline(self, num_tickets: int = 500) -> None:
        """
        Run the complete pipeline.

        Args:
            num_tickets: Number of tickets to generate
        """
        print("\n" + "="*70)
        print("SUPPORT TICKET CLASSIFICATION PIPELINE")
        print("="*70)

        try:
            # Step 1: Generate data
            self.step_1_generate_data(num_tickets)

            # Step 2: Preprocess
            self.step_2_preprocess_data()

            # Step 3: Train category models
            self.step_3_train_category_models()

            # Step 4: Train priority models
            self.step_4_train_priority_models()

            # Step 5: Generate predictions
            predictions = self.step_5_generate_predictions()

            # Step 6: Export results
            self.step_6_export_results(predictions)

            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)

        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {str(e)}")
            raise


if __name__ == "__main__":
    # Run the pipeline
    pipeline = SupportTicketPipeline(output_dir="../data")
    pipeline.run_full_pipeline(num_tickets=500)
