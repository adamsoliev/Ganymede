// https://github.com/benhoyt/ht

/*
        API
    ht_create
    ht_destroy
    ht_get
    ht_set
    ht_length
*/

#include "ganymede.h"

#define INITIAL_CAPACITY 16  // must not be zero & must be power of 2
#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

ht* ht_create(void) {
        // Allocate space for hash table struct.
        ht* table = malloc(sizeof(ht));
        if (table == NULL) {
                return NULL;
        }
        table->length = 0;
        table->capacity = INITIAL_CAPACITY;

        // Allocate (zero'd) space for entry buckets.
        table->entries = calloc(table->capacity, sizeof(ht_entry));
        if (table->entries == NULL) {
                free(table);  // error, free table before we return!
                return NULL;
        }
        return table;
}

void ht_destroy(ht* table) {
        // First free allocated keys.
        for (size_t i = 0; i < table->capacity; i++) {
                free((void*)table->entries[i].key);
        }

        // Then free entries array and table itself.
        free(table->entries);
        free(table);
}

static uint64_t hash_key(const char* key) {
        uint64_t hash = FNV_OFFSET;
        for (const char* p = key; *p; p++) {
                hash ^= (uint64_t)(unsigned char)(*p);
                hash *= FNV_PRIME;
        }
        return hash;
}

void* ht_get(ht* table, const char* key) {
        // AND hash with capacity-1 to ensure it's within entries array.
        uint64_t hash = hash_key(key);
        size_t index = (size_t)(hash & (uint64_t)(table->capacity - 1));

        // Loop till we find an empty entry.
        while (table->entries[index].key != NULL) {
                if (strcmp(key, table->entries[index].key) == 0) {
                        // Found key, return value.
                        return table->entries[index].value;
                }
                // Key wasn't in this slot, move to next (linear probing).
                index++;
                if (index >= table->capacity) {
                        // At end of entries array, wrap around.
                        index = 0;
                }
        }
        return NULL;
}

// Internal function to set an entry (without expanding table).
static const char* ht_set_entry(ht_entry* entries, size_t capacity, const char* key, void* value,
                                size_t* plength) {
        // AND hash with capacity-1 to ensure it's within entries array.
        uint64_t hash = hash_key(key);
        size_t index = (size_t)(hash & (uint64_t)(capacity - 1));

        // Loop till we find an empty entry.
        while (entries[index].key != NULL) {
                if (strcmp(key, entries[index].key) == 0) {
                        // Found key (it already exists), update value.
                        entries[index].value = value;
                        return entries[index].key;
                }
                // Key wasn't in this slot, move to next (linear probing).
                index++;
                if (index >= capacity) {
                        // At end of entries array, wrap around.
                        index = 0;
                }
        }

        // Didn't find key, allocate+copy if needed, then insert it.
        if (plength != NULL) {
                key = strdup(key);
                if (key == NULL) {
                        return NULL;
                }
                (*plength)++;
        }
        entries[index].key = (char*)key;
        entries[index].value = value;
        return key;
}

// Expand hash table to twice its current size. Return true on success,
// false if out of memory.
static bool ht_expand(ht* table) {
        // Allocate new entries array.
        size_t new_capacity = table->capacity * 2;
        if (new_capacity < table->capacity) {
                return false;  // overflow (capacity would be too big)
        }
        ht_entry* new_entries = calloc(new_capacity, sizeof(ht_entry));
        if (new_entries == NULL) {
                return false;
        }

        // Iterate entries, move all non-empty ones to new table's entries.
        for (size_t i = 0; i < table->capacity; i++) {
                ht_entry entry = table->entries[i];
                if (entry.key != NULL) {
                        ht_set_entry(new_entries, new_capacity, entry.key, entry.value, NULL);
                }
        }

        // Free old entries array and update this table's details.
        free(table->entries);
        table->entries = new_entries;
        table->capacity = new_capacity;
        return true;
}

const char* ht_set(ht* table, const char* key, void* value) {
        assert(value != NULL);

        // If length will exceed half of current capacity, expand it.
        if (table->length >= table->capacity / 2) {
                if (!ht_expand(table)) {
                        return NULL;
                }
        }

        // Set entry and update length.
        return ht_set_entry(table->entries, table->capacity, key, value, &table->length);
}

size_t ht_length(ht* table) { return table->length; }

hti ht_iterator(ht* table) {
        hti it;
        it._table = table;
        it._index = 0;
        return it;
}

bool ht_next(hti* it) {
        // Loop till we've hit end of entries array.
        ht* table = it->_table;
        while (it->_index < table->capacity) {
                size_t i = it->_index;
                it->_index++;
                if (table->entries[i].key != NULL) {
                        // Found next non-empty item, update iterator key and value.
                        ht_entry entry = table->entries[i];
                        it->key = entry.key;
                        it->value = entry.value;
                        return true;
                }
        }
        return false;
}

// Takes a printf-style format string and returns a formatted string.
char* format(char* fmt, ...) {
        char* buf;
        size_t buflen;
        FILE* out = open_memstream(&buf, &buflen);

        va_list ap;
        va_start(ap, fmt);
        vfprintf(out, fmt, ap);
        va_end(ap);
        fclose(out);
        return buf;
}

char* strdup(const char* s) {
        size_t slen = strlen(s);
        char* result = malloc(slen + 1);
        if (result == NULL) {
                return NULL;
        }

        memcpy(result, s, slen + 1);
        return result;
}

void ht_test(void) {
        printf("HT tests running...\n");
        ht* table = ht_create();
        for (int i = 1; i < 20001; i++) ht_set(table, format("key %d", i), (void*)(size_t)i);
        assert(ht_length(table) == 20000);
        for (int i = 20001; i < 35001; i++)
                ht_set(table, format("key %d -", i), (void*)(size_t)(i * 20));
        assert(ht_length(table) == 35000);
        for (int i = 35001; i < 50000; i++)
                ht_set(table, format("key %d - *", i), (void*)(size_t)(i * 300));
        assert(ht_length(table) == 49999);

        for (int i = 1; i < 20001; i++)
                assert(ht_get(table, format("key %d", i)) == (void*)(size_t)i);
        for (int i = 20001; i < 35001; i++)
                assert(ht_get(table, format("key %d -", i)) == (void*)(size_t)(i * 20));
        for (int i = 35001; i < 50000; i++)
                assert(ht_get(table, format("key %d - *", i)) == (void*)(size_t)(i * 300));

        for (int i = 1; i < 50000; i++) assert(ht_get(table, "no such key") == NULL);
        printf("OK\n");
}
