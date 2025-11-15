package ma.emsi.Chidoub.test4;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface Assistant {

    @SystemMessage("""
            Tu es un assistant conversationnel qui peut utiliser ou non le RAG.
            Si la question concerne l'intelligence artificielle ou le RAG, tu peux t'appuyer
            sur le support de cours fourni. Sinon, réponds normalement sans citer le document.
            Réponds en français, de façon claire et concise.
            """)
    @UserMessage("Question : {{it}}")
    String answer(String question);
}
