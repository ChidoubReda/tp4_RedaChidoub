package ma.emsi.Chidoub.test3;

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

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    public static void main(String[] args) {

        System.out.println("\n--- TEST 3 : Routage RAG (LangChain4j + Gemini) ---\n");

        configureLogger();

        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) {
            throw new IllegalStateException("⚠️ Définir GEMINI_API_KEY avant d'exécuter.");
        }

        // --- 1) Parser PDF / fichiers
        DocumentParser parser = new ApacheTikaDocumentParser();

        Document docIA = load("src/main/resources/rag.pdf", parser);
        Document docAutre = load("src/main/resources/autre.txt", parser);

        // --- 2) Splitter
        List<TextSegment> chunksIA = split(docIA);
        List<TextSegment> chunksAutre = split(docAutre);

        // --- 3) Embeddings (Google text-embedding-004)
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // --- 4) 2 EmbeddingStores (un pour chaque document)
        EmbeddingStore<TextSegment> storeIA = buildStore(chunksIA, embeddingModel);
        EmbeddingStore<TextSegment> storeAutre = buildStore(chunksAutre, embeddingModel);

        // --- 5) 2 ContentRetrievers
        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.2)
                .build();

        ContentRetriever retrieverAutre = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeAutre)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.2)
                .build();


        // --- 6) ChatModel (Gemini) avec logging
        ChatModel llm = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .timeout(Duration.ofSeconds(60))
                .logRequestsAndResponses(true)
                .build();

        // --- 7) Description des sources (Map<retriever, description>)
        Map<ContentRetriever, String> routes = new HashMap<>();
        routes.put(retrieverIA,
                "Documents sur l'intelligence artificielle, le RAG, et l'analyse de texte.");
        routes.put(retrieverAutre,
                "Documents qui parlent d'un sujet non lié à l'intelligence artificielle.");

        // --- 8) QueryRouter (utilise le LLM pour choisir la bonne route)
        QueryRouter router = new LanguageModelQueryRouter(llm, routes);

        // --- 9) RetrievalAugmentor
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // --- 10) Assistant RAG avec routage
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(llm)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // --- 11) Console interactive
        try (Scanner sc = new Scanner(System.in)) {
            System.out.println("Pose une question (exit pour quitter) :");
            while (true) {
                System.out.print("> ");
                String q = sc.nextLine().trim();
                if (q.equalsIgnoreCase("exit")) break;
                if (q.isEmpty()) continue;

                System.out.println("\n🤖 " + assistant.answer(q) + "\n");
            }
        }
    }

    // --------------------------
    // Méthodes utilitaires
    // --------------------------

    private static Document load(String path, DocumentParser parser) {
        return FileSystemDocumentLoader.loadDocument(Paths.get(path), parser);
    }

    private static List<TextSegment> split(Document document) {
        return DocumentSplitters.recursive(200, 40).split(document);
    }

    private static EmbeddingStore<TextSegment> buildStore(List<TextSegment> chunks,
                                                          EmbeddingModel model) {
        List<Embedding> vectors = model.embedAll(chunks).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, chunks);
        return store;
    }

    private static void configureLogger() {
        Logger log = Logger.getLogger("dev.langchain4j");
        log.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        if (log.getHandlers().length == 0) {
            log.addHandler(handler);
        }
    }
}
