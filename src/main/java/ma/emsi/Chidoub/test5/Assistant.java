package ma.emsi.Chidoub.test5;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface Assistant {

    @SystemMessage("""
            Tu es un assistant RAG qui combine un support de cours sur le RAG
            et des recherches sur le Web. Utilise le document PDF quand c'est pertinent,
            et complète avec des informations du Web si nécessaire.
            Réponds en français, de façon claire et structurée.
            """)
    @UserMessage("Question : {{it}}")
    String answer(String question);
}