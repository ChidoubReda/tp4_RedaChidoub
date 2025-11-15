package ma.emsi.Chidoub.test3;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface Assistant {

    @SystemMessage("""
            Tu es un assistant utilisant un système RAG multi-sources.
            Ton rôle est d'analyser la question et de sélectionner le bon document via le routage.
            Réponds de manière claire, concise et uniquement à partir des documents disponibles.
            """)
    @UserMessage("Question : {{it}}")
    String answer(String question);
}
