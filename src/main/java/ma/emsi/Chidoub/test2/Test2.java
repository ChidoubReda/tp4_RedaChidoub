package ma.emsi.Chidoub.test2;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.Scanner;

// Pour la configuration du logger
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test2 {

    public static void main(String[] args) {

        System.out.println("\n--- TEST 2 : RAG + Logging (Gemini) ---\n");

        // 1) Activer le logging détaillé pour LangChain4j
        configureLogger();

        // 2) Charger le document source (PDF)
        DocumentParser parser = new ApacheTikaDocumentParser();
        Path pdfPath = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        // 3) Découper le document en segments (chunks)
        var splitter = DocumentSplitters.recursive(250, 40);
        List<TextSegment> segments = splitter.split(document);

        // 4) Initialiser le modèle d'embedding (Google text-embedding-004)
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) {
            throw new IllegalStateException("La variable d'environnement GEMINI_API_KEY n'est pas définie.");
        }

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // 5) Calculer les embeddings pour tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 6) Stocker les embeddings dans une base vectorielle en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // 7) Configurer le modèle de chat Gemini avec logging des requêtes/réponses
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash")
                .temperature(0.2)                   // température basse pour le RAG
                .timeout(Duration.ofSeconds(60))
                .logRequestsAndResponses(true)
                .build();

        // 8) Récupérateur de contenu (RAG)
        var contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.3)
                .build();

        // 9) Mémoire de conversation (on garde les derniers échanges)
        var chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // 10) Création de l’assistant RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();

        // 11) Boucle interactive en console
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Pose une question sur le PDF (tape 'exit' pour quitter) :");
            while (true) {
                System.out.print("> ");
                String userInput = scanner.nextLine().trim();

                if (userInput.equalsIgnoreCase("exit")) {
                    System.out.println("Fin du test 2.");
                    break;
                }
                if (userInput.isEmpty()) {
                    continue;
                }

                String answer = assistant.reply(userInput);
                System.out.println("\n🤖 " + answer + "\n");
            }
        }
    }

    /**
     * Configure le logger sous-jacent pour afficher les logs détaillés de LangChain4j.
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        // Éviter d'ajouter plusieurs fois le même handler si main est relancé
        if (packageLogger.getHandlers().length == 0) {
            packageLogger.addHandler(handler);
        }
    }
}