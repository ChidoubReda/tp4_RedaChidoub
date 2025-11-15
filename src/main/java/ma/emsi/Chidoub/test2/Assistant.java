package ma.emsi.Chidoub.test2;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface Assistant {

    @SystemMessage("""
            Tu es un assistant basé sur un système RAG.
            Tes réponses doivent se fonder uniquement sur le contenu du PDF fourni.
            Explique de manière claire et structurée, en français.
            Si une information ne se trouve pas dans le document, indique-le honnêtement.
            """)
    @UserMessage("Question de l'utilisateur : {{it}}")
    String reply(String question);
}
