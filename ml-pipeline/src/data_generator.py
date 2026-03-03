"""
Support Ticket Dataset Generator

Generates synthetic support tickets with realistic categories and priorities
for ML model training and evaluation.

Categories: Billing, Technical Issue, Account, General Query
Priorities: High, Medium, Low
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Seed for reproducibility
random.seed(42)


class SupportTicketGenerator:
    """Generate synthetic support tickets for ML training."""

    # Ticket templates by category
    TICKET_TEMPLATES = {
        "Billing": {
            "keywords": ["charge", "invoice", "payment", "refund", "subscription", "cost", "price", "billing", "credit card", "transaction"],
            "templates": [
                "I was charged twice for my subscription this month. Can you help?",
                "Why is my invoice showing a different amount than expected?",
                "I need to update my payment method urgently.",
                "Can I get a refund for the last month's charges?",
                "My subscription renewal failed. Please retry the payment.",
                "The pricing seems incorrect on my latest bill.",
                "I was overcharged for the premium plan upgrade.",
                "Can you explain the additional charges on my account?",
                "I want to cancel my subscription and get a refund.",
                "My credit card was declined. Can you help process the payment?",
            ]
        },
        "Technical Issue": {
            "keywords": ["error", "bug", "crash", "not working", "broken", "issue", "problem", "fail", "slow", "timeout"],
            "templates": [
                "The application keeps crashing when I try to upload files.",
                "I'm getting a 500 error on the dashboard page.",
                "The search functionality is not working properly.",
                "The system is running extremely slowly today.",
                "I can't connect to the API endpoint.",
                "The data export feature is broken.",
                "I'm experiencing timeout errors when loading reports.",
                "The mobile app won't sync with the desktop version.",
                "The database connection seems to be failing intermittently.",
                "The authentication system is rejecting valid credentials.",
            ]
        },
        "Account": {
            "keywords": ["account", "login", "password", "reset", "access", "permission", "user", "profile", "two-factor", "mfa"],
            "templates": [
                "I forgot my password and can't reset it.",
                "I need to add another user to my team account.",
                "Can you help me recover my account? I can't log in.",
                "I want to change my account email address.",
                "How do I enable two-factor authentication?",
                "I need to update my profile information.",
                "Can you remove a team member from my account?",
                "I'm locked out of my account after multiple login attempts.",
                "How do I transfer my account to a different email?",
                "I need to change my account permissions for a user.",
            ]
        },
        "General Query": {
            "keywords": ["how", "what", "when", "where", "why", "help", "guide", "tutorial", "documentation", "feature"],
            "templates": [
                "How do I use the advanced filtering options?",
                "What are the system requirements for your software?",
                "Can you explain how the reporting feature works?",
                "Where can I find the API documentation?",
                "Is there a tutorial for the new dashboard?",
                "How do I export data in different formats?",
                "What's the best way to organize my projects?",
                "Can you recommend the best practices for using your platform?",
                "How do I integrate your service with my existing tools?",
                "Where can I find information about upcoming features?",
            ]
        }
    }

    # Priority indicators
    PRIORITY_INDICATORS = {
        "High": ["urgent", "critical", "asap", "immediately", "emergency", "down", "broken", "can't work", "blocked", "severe"],
        "Medium": ["issue", "problem", "error", "help", "need", "please", "soon", "important"],
        "Low": ["question", "how", "what", "curious", "wondering", "suggestion", "feedback", "general"]
    }

    @staticmethod
    def generate_dataset(num_tickets: int = 500) -> Tuple[List[Dict], Dict]:
        """
        Generate synthetic support tickets.

        Args:
            num_tickets: Number of tickets to generate

        Returns:
            Tuple of (tickets list, statistics dict)
        """
        tickets = []
        category_counts = {cat: 0 for cat in SupportTicketGenerator.TICKET_TEMPLATES.keys()}
        priority_counts = {"High": 0, "Medium": 0, "Low": 0}

        for i in range(num_tickets):
            # Select category with weighted distribution
            category = random.choices(
                list(SupportTicketGenerator.TICKET_TEMPLATES.keys()),
                weights=[0.25, 0.35, 0.20, 0.20],
                k=1
            )[0]

            # Generate ticket text
            template = random.choice(SupportTicketGenerator.TICKET_TEMPLATES[category]["templates"])
            text = template

            # Determine priority based on keywords
            priority = SupportTicketGenerator._determine_priority(text, category)

            # Create ticket
            ticket = {
                "id": f"TICKET-{1000 + i}",
                "text": text,
                "category": category,
                "priority": priority,
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                "status": random.choice(["Open", "In Progress", "Resolved", "Closed"])
            }

            tickets.append(ticket)
            category_counts[category] += 1
            priority_counts[priority] += 1

        # Calculate statistics
        stats = {
            "total_tickets": num_tickets,
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
            "categories": list(category_counts.keys()),
            "priorities": ["High", "Medium", "Low"]
        }

        return tickets, stats

    @staticmethod
    def _determine_priority(text: str, category: str) -> str:
        """
        Determine priority based on text content and category.

        Args:
            text: Ticket text
            category: Ticket category

        Returns:
            Priority level: High, Medium, or Low
        """
        text_lower = text.lower()

        # Check for high priority indicators
        for keyword in SupportTicketGenerator.PRIORITY_INDICATORS["High"]:
            if keyword in text_lower:
                return "High"

        # Technical issues are often medium priority
        if category == "Technical Issue":
            return random.choice(["High", "Medium", "Medium"])

        # Billing issues are often medium priority
        if category == "Billing":
            return random.choice(["Medium", "Medium", "Low"])

        # Check for medium priority indicators
        for keyword in SupportTicketGenerator.PRIORITY_INDICATORS["Medium"]:
            if keyword in text_lower:
                return "Medium"

        # Default to low priority
        return "Low"

    @staticmethod
    def save_dataset(tickets: List[Dict], filepath: str) -> None:
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(tickets, f, indent=2)

    @staticmethod
    def load_dataset(filepath: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    # Generate and save dataset
    tickets, stats = SupportTicketGenerator.generate_dataset(num_tickets=500)
    SupportTicketGenerator.save_dataset(tickets, "support_tickets.json")

    print("Dataset Generated Successfully!")
    print(f"Total Tickets: {stats['total_tickets']}")
    print("\nCategory Distribution:")
    for cat, count in stats['category_distribution'].items():
        print(f"  {cat}: {count} ({count/stats['total_tickets']*100:.1f}%)")
    print("\nPriority Distribution:")
    for pri, count in stats['priority_distribution'].items():
        print(f"  {pri}: {count} ({count/stats['total_tickets']*100:.1f}%)")
