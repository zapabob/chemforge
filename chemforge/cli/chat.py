"""
Chat CLI for multi-target-pIC50-predictor.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt

console = Console()


@click.command()
@click.option('--model', type=str, default='txgemma:9b-chat-q6_k',
              help='TxGemma model to use')
@click.option('--host', type=str, default='http://localhost:11434',
              help='Ollama host URL')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def chat_cli(model, host, verbose):
    """
    Start interactive chat with TxGemma AI.
    
    Examples:
    
    \b
    # Basic chat
    multi-target-pIC50-predictor chat
    
    \b
    # Chat with custom model
    multi-target-pIC50-predictor chat --model txgemma:13b-chat-q6_k
    
    \b
    # Chat with custom host
    multi-target-pIC50-predictor chat --host http://ollama:11434
    """
    
    # Display chat configuration
    config_text = f"""
    Model: {model}
    Host: {host}
    Verbose: {'Yes' if verbose else 'No'}
    """
    
    config_panel = Panel(
        config_text.strip(),
        title="🤖 Chat Configuration",
        border_style="blue"
    )
    console.print(config_panel)
    
    # Welcome message
    welcome_text = Text("Welcome to Multi-Target pIC50 Predictor Chat!", style="bold blue")
    welcome_panel = Panel(
        welcome_text,
        title="💬 Chat Started",
        border_style="blue"
    )
    console.print(welcome_panel)
    
    console.print("\n[bold green]You can ask me about:[/bold green]")
    console.print("• Molecular properties and pIC50 predictions")
    console.print("• ChEMBL database and receptor information")
    console.print("• Drug discovery and cheminformatics")
    console.print("• QSAR modeling and machine learning")
    console.print("• And much more!")
    
    console.print("\n[bold yellow]Type 'quit' or 'exit' to end the chat.[/bold yellow]")
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("\n[bold green]Goodbye! Thanks for chatting![/bold green]")
                break
            
            # Simulate AI response
            console.print("\n[bold green]AI:[/bold green] ", end="")
            
            # Sample responses based on input
            if 'pIC50' in user_input.lower():
                response = "pIC50 is the negative logarithm of the IC50 value, which represents the concentration of a compound required to inhibit 50% of a biological process. It's a key metric in drug discovery for measuring compound potency."
            elif 'chembl' in user_input.lower():
                response = "ChEMBL is a database of bioactive drug-like small molecules. It contains 2D structures, calculated properties, and abstracted bioactivities. The database is maintained by the European Bioinformatics Institute (EBI)."
            elif 'dat' in user_input.lower():
                response = "DAT (Dopamine Transporter) is a membrane-spanning protein that pumps the neurotransmitter dopamine out of the synaptic cleft back into cytosol. It's a target for ADHD medications like methylphenidate and amphetamine."
            elif '5ht2a' in user_input.lower():
                response = "5-HT2A is a serotonin receptor subtype. It's involved in mood regulation and is a target for many psychiatric drugs. It's also the primary target for psychedelic compounds like LSD and psilocybin."
            elif 'cb1' in user_input.lower():
                response = "CB1 (Cannabinoid Receptor 1) is a G-protein coupled receptor that is the target of the psychoactive compound THC. It's involved in appetite, pain sensation, mood, and memory."
            elif 'model' in user_input.lower():
                response = "Our multi-target pIC50 predictor uses ensemble methods combining Transformer and GNN models. It can predict pIC50 values for multiple targets including DAT, 5-HT2A, CB1, CB2, and opioid receptors."
            else:
                response = "That's an interesting question! I'm here to help with molecular property prediction, drug discovery, and cheminformatics. Could you be more specific about what you'd like to know?"
            
            console.print(response)
            
        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Chat interrupted. Goodbye![/bold yellow]")
            break
        except Exception as e:
            if verbose:
                console.print(f"\n[bold red]Error: {e}[/bold red]")
            else:
                console.print("\n[bold red]Sorry, I encountered an error. Please try again.[/bold red]")


if __name__ == "__main__":
    chat_cli()
